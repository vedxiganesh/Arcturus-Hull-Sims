"""Local <-> cluster transfers. The only code that runs ssh.

  push_code()            hullsweep code -> ~/hullsweep_code (atomic swap) + VERSION
  push_topology(name)    hullsweep_data/<name>/{topology.json, template[, meshes]}
                         -> ~/orcd/pool/hullsweep/<name>/
  pull(study)            the study's small files -> hullsweep_data/<topo>/<study>/
                         (optionally the final .cas/.dat of each chain's result)

Each transfer is ONE ssh connection carrying a tar stream (one Duo prompt, no
rsync needed on Windows). Remote paths come from layout.py; nothing here is
generated per study and nothing is kept in two places.
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import layout

#: Code paths never shipped to the cluster.
CODE_EXCLUDE = ("sweeps/", "introspection/", "tests/", "topologies/", "templates/", ".git/")
TEXT_SUFFIXES = (".py", ".sh", ".md", ".json", ".in", ".txt", ".csv")
EXECUTABLE = ("cluster/case_job.sh", "bin/hs")


def _require_local() -> None:
    if layout.site() != "local":
        sys.exit("sync runs on the local machine (it pushes to / pulls from the cluster)")


def _ssh(script: str, stdin=None, stdout=None, host: str = layout.REMOTE_HOST):
    """Run a bash script on the cluster in one ssh connection."""
    cmd = ["ssh", "-o", "ServerAliveInterval=30", host, "bash -c " + shlex.quote(script)]
    return subprocess.Popen(cmd, stdin=stdin, stdout=stdout)


def _finish(proc, what: str) -> None:
    rc = proc.wait()
    if rc != 0:
        sys.exit(f"{what} failed on the cluster (exit {rc}); see the output above")


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


def code_files() -> list[str]:
    """Tracked + untracked-but-not-ignored files, minus CODE_EXCLUDE."""
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                       cwd=layout.CODE_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git ls-files failed in {layout.CODE_DIR}: {r.stderr.strip()}")
    files = sorted({f for f in r.stdout.splitlines()
                    if f and not f.startswith(CODE_EXCLUDE) and (layout.CODE_DIR / f).is_file()})
    for need in ("hs.py", "layout.py", "run_case.py", *EXECUTABLE):
        if need not in files:
            sys.exit(f"{need} is missing from the code file list (ignored by git?)")
    return files


def _read_code(rel: str) -> bytes:
    data = (layout.CODE_DIR / rel).read_bytes()
    if rel.endswith(TEXT_SUFFIXES) or rel in EXECUTABLE:
        data = data.replace(b"\r\n", b"\n")  # bash on the cluster chokes on CRLF
    return data


def version_info(files: list[str]) -> dict:
    h = hashlib.sha1()
    for f in files:
        h.update(f.encode() + b"\0" + _read_code(f) + b"\0")

    def git(*a):
        r = subprocess.run(["git", *a], cwd=layout.CODE_DIR, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"content_sha1": h.hexdigest(), "commit": git("rev-parse", "--short", "HEAD") or "no-commits",
            "dirty": bool(git("status", "--porcelain")), "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_files": len(files)}


def code_tarball(files: list[str], version: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in [*files, "VERSION"]:
            data = (json.dumps(version, indent=1) + "\n").encode() if rel == "VERSION" else _read_code(rel)
            info = tarfile.TarInfo(rel)
            info.size, info.mtime = len(data), int(time.time())
            info.mode = 0o755 if rel in EXECUTABLE else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def push_code(dry_run: bool = False) -> dict:
    _require_local()
    files = code_files()
    version = version_info(files)
    print(f"{len(files)} files, content {version['content_sha1'][:10]}, commit {version['commit']}"
          f"{' (dirty)' if version['dirty'] else ''}")
    if dry_run:
        print("\n".join(f"  {f}" for f in files))
        return version
    dest = layout.remote_code().replace("~", "$HOME", 1)
    # $HOME is NFS: a script a running job still has open (case_job.sh, bin/hs)
    # leaves a .nfs* file behind when deleted, and its directory cannot be
    # removed. Busy leftovers are renamed aside, never allowed to abort the push.
    script = f"""set -e
dest="{dest}"; new="$dest.new"; old="$dest.old"; stamp=$(date +%s)
for d in "$new" "$old"; do
    if [ -e "$d" ]; then rm -rf "$d" 2>/dev/null || mv "$d" "$d.busy.$stamp"; fi
done
rm -rf "$dest".*.busy.* 2>/dev/null || true
mkdir -p "$new"
tar xzf - -C "$new"
if [ -d "$dest" ]; then mv "$dest" "$old"; fi
mv "$new" "$dest"
rm -rf "$old" 2>/dev/null || echo "note: $old is held by running jobs; it is cleared on a later push"
echo "REMOTE_CONTENT $(sed -n 's/.*"content_sha1": "\\([0-9a-f]*\\)".*/\\1/p' "$dest/VERSION")"
"""
    p = _ssh(script, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = p.communicate(code_tarball(files, version))
    text = out.decode(errors="replace")
    print(text.rstrip())
    if p.returncode != 0:
        sys.exit(f"push-code failed on the cluster (exit {p.returncode}); the cluster still runs the old code")
    got = next((ln.split()[1] for ln in text.splitlines() if ln.startswith("REMOTE_CONTENT ") and
                len(ln.split()) > 1), None)
    if got != version["content_sha1"]:
        sys.exit(f"push-code did NOT take: cluster reports content {got}, local is "
                 f"{version['content_sha1'][:10]}")
    print(f"verified: {layout.remote_code()} is content {got[:10]} (commit {version['commit']})")
    return version


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


def push_topology(name: str, with_meshes: bool = False, force: bool = False) -> None:
    """Upload a topology. Refuses to change files that existing studies depend on."""
    _require_local()
    topo = layout.load_topology(name)
    if topo.get("template_hull_stats") is None:
        sys.exit(f"{name}: no template_hull_stats in topology.json; run `prepare_case.py template` first")
    files = [layout.topology_dir(name) / layout.TOPOLOGY_FILE, layout.topology_file(name, "template")]
    if with_meshes:
        files += [p for p in (layout.topology_file(name, "foreground_mesh", required=False),
                              layout.topology_file(name, "background_mesh", required=False)) if p]
    total = sum(p.stat().st_size for p in files)
    print(f"pushing {len(files)} files ({total / 1e6:.0f} MB) to {layout.remote_pool(name)}:")
    for p in files:
        print(f"  {p.name}")

    dest = layout.remote_pool(name).replace("~", "$HOME", 1)
    tmpl_glob = layout.ROLE_PATTERNS["template"].format(name=name)
    script = f"""set -e
D="{dest}"; I="$D.incoming"
rm -rf "$I"; mkdir -p "$I"
tar xf - -C "$I"
nstudies=$(find "$D" -mindepth 2 -maxdepth 2 -name study.json 2>/dev/null | wc -l)
if [ "$nstudies" -gt 0 ] && [ "{int(force)}" != 1 ]; then
    for f in $(ls "$I"); do
        if [ -e "$D/$f" ] && ! cmp -s "$I/$f" "$D/$f"; then
            echo "REFUSED: $f differs from the copy $nstudies existing studies were created with" >&2
            echo "(create a new topology name, or --force if no study will run again)" >&2
            rm -rf "$I"; exit 3
        fi
    done
fi
mkdir -p "$D"
for f in $(ls "$I"); do
    case "$f" in {tmpl_glob})
        for old in "$D"/{tmpl_glob}; do [ -e "$old" ] && [ "$(basename "$old")" != "$f" ] && rm -v "$old"; done;;
    esac
    mv -f "$I/$f" "$D/$f"
done
rmdir "$I"
ls -la "$D"
"""
    p = _ssh(script, stdin=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdin, mode="w|") as tar:
        for f in files:
            tar.add(str(f), arcname=f.name)
    p.stdin.close()
    _finish(p, "push-topology")


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


def _extract_stream(stream, dest: Path) -> list[str]:
    names = []
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"unsafe path in tar stream: {m.name}")
            if hasattr(tarfile, "data_filter"):
                tar.extract(m, dest, filter="data")
            else:
                tar.extract(m, dest)
            if m.isfile():
                names.append(m.name)
    return names


def pull(study: str, with_data: bool = False) -> Path:
    """Study files from pool + scratch into hullsweep_data/<topo>/<study>/ (one merged tree)."""
    _require_local()
    layout.check_name(study, "study")
    pool = layout.remote_pool().replace("~", "$HOME", 1)
    scr = layout.remote_scratch().replace("~", "$HOME", 1)
    script = f"""set -e
P="{pool}"; S="{scr}"; N="{study}"
cd "$P"
hits=$(ls -d */"$N"/study.json 2>/dev/null || true)
n=$(printf "%s" "$hits" | grep -c . || true)
if [ "$n" != 1 ]; then echo "study $N found $n times under $P: $hits" >&2; exit 3; fi
T="${{hits%%/*}}"
echo "$T" >&2
args=(-C "$P" --exclude='*.tmp' --exclude='ledger.lock' "$T/$N")
if [ -d "$S/$T/$N" ]; then
    cd "$S"
    mapfile -t F < <(find "$T/$N" -maxdepth 2 -type f \\( -name '*.json' -o -name '*.out' -o -name '*.6dof' -o -name 'sw-motion.csv' -o -name 'sw_sdof.c' -o -name '{layout.PROGRESS_FILE}' \\))
    [ "${{#F[@]}}" -gt 0 ] && args+=(-C "$S" "${{F[@]}}")
fi
tar czf - "${{args[@]}}"
"""
    dest = layout.LOCAL_DATA
    dest.mkdir(parents=True, exist_ok=True)
    p = _ssh(script, stdout=subprocess.PIPE)
    names = _extract_stream(p.stdout, dest)
    _finish(p, "pull")
    if not names:
        sys.exit("nothing pulled")
    topo = Path(names[0]).parts[0]
    local = dest / topo / study
    print(f"pulled {len(names)} files into {local}")

    if with_data:
        led = json.loads((local / layout.LEDGER_FILE).read_text())
        cids = set()
        for ch in led["chains"].values():  # chain results that live in THIS study's scratch
            res = ch.get("result") or {}
            cids |= {c for c in (res.get("case_id"), (res.get("free") or {}).get("case_id")) if c in led["cases"]}
        cids = sorted(cids)
        if not cids:
            print("no chain results yet; no data to pull")
            return local
        files = " ".join(shlex.quote(f"{topo}/{study}/{c}/{c}_final.{ext}")
                         for c in cids for ext in ("cas.h5", "dat.h5"))
        p = _ssh(f'set -e; cd "{scr}"; tar czf - {files}', stdout=subprocess.PIPE)
        got = _extract_stream(p.stdout, dest)
        _finish(p, "pull --with-data")
        print(f"pulled {len(got)} data files for {', '.join(cids)}")
    return local
