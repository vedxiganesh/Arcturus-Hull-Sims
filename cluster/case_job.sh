#!/bin/bash -l
# One CFD case of a hullsweep study. Submitted by the orchestrator (ledger.py):
#
#   sbatch <resources> ~/hullsweep_code/cluster/case_job.sh <study> <case_id>
#
# Static: nothing in here is generated or edited per study. Resources come from
# the sbatch command line (study.json), paths from layout.py via `hs.py path`.
#
# Walltime: run_case.py stops stop_margin_s before the job's end, writes its
# data and reports "incomplete"; this script then requeues the job, which
# resumes in the same per-case directory. Preemption requeues via --requeue.
# `-l` (login shell) so `module` works under a job submitted from another job.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: case_job.sh <study> <case_id>" >&2
    exit 2
fi
export STUDY="$1"
export CASE_ID="$2"
export CODE="$HOME/hullsweep_code"
MAX_RESTARTS=12

source "$HOME/ansys_env.sh"

module load miniforge
set +u
conda activate pyfluent
set -u

CASE_DIR=$(python "$CODE/hs.py" path case-dir --study "$STUDY" --case "$CASE_ID")
mkdir -p "$CASE_DIR"

echo "job      : ${SLURM_JOB_ID} (restart count ${SLURM_RESTART_COUNT:-0})"
echo "node     : $SLURM_JOB_NODELIST   ntasks: $SLURM_NTASKS"
echo "study    : $STUDY"
echo "case     : $CASE_ID"
echo "case dir : $CASE_DIR"

# --- Licensing: Shared Web via CA-bundle bind mount (from submit_hull.sh) ---
# LOAD-BEARING. ansyscl ignores SSL_CERT_FILE and reads
# /etc/pki/tls/certs/ca-bundle.crt, which on Rocky 8 lacks Sectigo Root R46.
export CA_BUNDLE="$HOME/ansys_ca_bundle.pem"
export CA_TARGET="/etc/pki/tls/certs/ca-bundle.crt"
export CA_EXPECTED_CERTS=289
[ -r "$CA_BUNDLE" ] || { echo "FATAL: merged CA bundle not found: $CA_BUNDLE" >&2; exit 1; }

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    echo "FATAL: multi-node unsupported (mount namespace is per node)." >&2
    exit 1
fi

END_ISO=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
export HULL_DEADLINE_EPOCH=$(date -d "$END_ISO" +%s)
echo "deadline : $END_ISO"

cd "$CASE_DIR"

rc=0
unshare -r --mount bash -c '
    set -euo pipefail
    mount --bind "$CA_BUNDLE" "$CA_TARGET"
    ns_certs=$(grep -c "BEGIN CERTIFICATE" "$CA_TARGET")
    if [ "$ns_certs" -ne "$CA_EXPECTED_CERTS" ]; then
        echo "FATAL: bind mount did not take ($ns_certs certs)." >&2
        exit 1
    fi
    exec python "$CODE/run_case.py" --study "$STUDY" --case "$CASE_ID"
' || rc=$?

STATE=$(python -c "import json;print(json.load(open('$CASE_DIR/status.json'))['state'])" 2>/dev/null || echo unknown)
echo "state    : $STATE (exit $rc)"
if [ "$STATE" = "incomplete" ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -ge "$MAX_RESTARTS" ]; then
        echo "FATAL: $MAX_RESTARTS restarts reached; not requeueing (the orchestrator will retry)." >&2
        exit 1
    fi
    echo "requeueing ${SLURM_JOB_ID} to continue"
    scontrol requeue "${SLURM_JOB_ID}"
fi

echo "done: $(date)"
exit "$rc"
