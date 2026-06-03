#!/bin/bash
# ============================================================================
# hpc_babysit.sh — SHARED-SERVER-SAFE status + result-retrieve babysitter
# ----------------------------------------------------------------------------
# This runs on a SHARED research-lab login node. Many people use this box and
# the cluster. It MUST stay polite. Read the HARD RULES before editing.
#
# HARD RULES (enforced in code below; do NOT relax them):
#   * READ-ONLY on shared infra. The only things this touches off-box are:
#       - `qstat -u nishioka`                  (read job state, nothing else)
#       - `ssh vancouver01 'ls -t ...'`        (read-only listing of FINISHED files)
#       - `scp` of ALREADY-FINISHED result files back to this host
#     NOTHING here submits, schedules, or mutates remote state.
#   * NEVER `qsub`/`bsub`/`sbatch` or otherwise submit/launch a job.
#   * NEVER launch GPU compute (no remote `python`, no `nohup ... &` on a node).
#   * NEVER `git stash` / `git pull` / `git fetch` / `git reset` a shared
#     checkout. The only git writes are: add/commit/push of THIS repo's own
#     newly-landed result files (the user's own nife repo / fork).
#   * NEVER run heavy compute on the login node. The one local computation is
#     aggregate_diffusion_sweep.py, which is a few-millisecond pandas tabulation
#     of tiny JSON files — and it runs under `nice`/`ionice` anyway.
#   * SINGLE PASS. No `while` loops, no `sleep`-poll, no daemonizing. cron is
#     the scheduler; this script does exactly one sweep and exits.
#   * Every `ssh`/`scp` is wrapped in `timeout` so a hung/unreachable host can
#     never wedge the cron slot. Host down => log it and exit 0 cleanly.
#   * Idempotent: re-running with no new state does nothing and commits nothing.
#
# WHAT IT DOES (one pass):
#   (1) STATUS: write the running PBS job count to /tmp/hpc_jobcount and append
#       a timestamped line to ~/nife_babysit.log.
#   (2) DIFFUSION SWEEP LANDING: if NO jobs are running AND new
#       results/diffusion_fit/D_fit_*.json have appeared since the last commit,
#       run aggregate_diffusion_sweep.py (local, light), then git add the new
#       D_fit_*.json + sweep_summary.csv, commit, and push (own repo only).
#   (3) GPU RETRIEVE: read-only `ssh vancouver01 'ls -t <results>'`, scp back any
#       result file newer than the local copy, and commit them.
#
# USAGE / SCHEDULING:
#   Run it by hand for a one-off check, or let cron run it. It is designed for a
#   GENTLE once-daily run. Do NOT add it to cron more aggressively than daily on
#   a shared box. Claude must NOT edit crontab — the USER installs it themselves
#   via `crontab -e`, adding exactly this line (gentle, once-daily at 07:00):
#
#     0 7 * * * /bin/bash /home/nishioka/IKM_Hiwi/nife/scripts/hpc_babysit.sh >> ~/nife_babysit.log 2>&1
#
# ============================================================================

# --- safety: do NOT `set -e`. A non-zero from one optional step (host down,
#     nothing to commit) must not abort the rest of the pass. We always exit 0.
set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Config (kept explicit; matches repo + cluster conventions)
# ---------------------------------------------------------------------------
REPO_ROOT="/home/nishioka/IKM_Hiwi/nife"
PBS_USER="nishioka"
JOBCOUNT_FILE="/tmp/hpc_jobcount"
LOG_FILE="${HOME}/nife_babysit.log"

DIFFUSION_DIR="results/diffusion_fit"                 # repo-relative
AGG_SCRIPT="scripts/analysis/aggregate_diffusion_sweep.py"
SWEEP_SUMMARY="${DIFFUSION_DIR}/sweep_summary.csv"

GPU_HOST="vancouver01"
GPU_RESULTS_DIR="/home/nishioka/nife_gpurun/results"  # remote, read-only listing
GPU_GLOB="*.json *.png"                               # finished-result patterns
LOCAL_GPU_DIR="results/_gpu_retrieve"                 # where scp lands them

# Be a good neighbour with timeouts (seconds) and low priority on local bits.
SSH_TIMEOUT=20
SCP_TIMEOUT=60
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
NICE="nice -n 19 ionice -c3"                          # idle CPU + idle I/O class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >>"${LOG_FILE}"; }

# Always land in the repo root (required; bare-import scripts + repo-relative paths).
cd "${REPO_ROOT}" || { echo "[$(ts)] FATAL: cannot cd ${REPO_ROOT}" >>"${LOG_FILE}" 2>/dev/null; exit 0; }

log "babysit: start (host=$(hostname -s 2>/dev/null) pid=$$)"

# ---------------------------------------------------------------------------
# (1) STATUS — count running PBS jobs (READ-ONLY qstat), record it.
# ---------------------------------------------------------------------------
# `qstat -u <user>` lists this user's jobs. We count lines whose state column is
# 'R' (running). If qstat is missing or errors, we record -1 and carry on; we do
# NOT treat "unknown" as "no jobs" (that gate guards the auto-commit below).
RUNNING=-1
if command -v qstat >/dev/null 2>&1; then
    # Read-only. awk picks the PBS state column 'R'. Tolerate empty output.
    RUNNING=$(qstat -u "${PBS_USER}" 2>/dev/null \
        | awk 'NR>5 && $10=="R"' \
        | wc -l \
        | tr -d ' ')
    # Guard against a non-numeric result from a weird qstat format.
    case "${RUNNING}" in
        ''|*[!0-9]*) RUNNING=-1 ;;
    esac
else
    log "status: qstat not found on PATH — skipping job-state checks"
fi

# Total of this user's jobs (any state) for context in the log line.
TOTAL=-1
if command -v qstat >/dev/null 2>&1; then
    TOTAL=$(qstat -u "${PBS_USER}" 2>/dev/null | awk 'NR>5 && $1 ~ /^[0-9]/' | wc -l | tr -d ' ')
    case "${TOTAL}" in ''|*[!0-9]*) TOTAL=-1 ;; esac
fi

# Write the running count to /tmp (best-effort; never fatal).
echo "${RUNNING}" >"${JOBCOUNT_FILE}" 2>/dev/null \
    || log "status: WARN could not write ${JOBCOUNT_FILE}"
log "status: running=${RUNNING} total=${TOTAL} (qstat -u ${PBS_USER})"

# ---------------------------------------------------------------------------
# (2) DIFFUSION SWEEP LANDING — only when the cluster is idle for this user.
# ---------------------------------------------------------------------------
# Rationale for the gate: if jobs are still running they may be writing fresh
# D_fit_*.json; committing mid-flight would capture half a sweep. We require a
# CONFIRMED zero (RUNNING==0). Unknown (-1) is treated as "do not touch".
if [ "${RUNNING}" = "0" ]; then
    # New = D_fit_*.json that git does not yet track/record as committed.
    # `git status --porcelain` lists untracked (??) and modified (' M') files.
    NEW_DFIT=$(git status --porcelain -- "${DIFFUSION_DIR}/D_fit_"*.json 2>/dev/null \
        | awk '{print $2}')

    if [ -n "${NEW_DFIT}" ]; then
        log "sweep: jobs idle + new D_fit json detected:"
        # shellcheck disable=SC2001
        echo "${NEW_DFIT}" | sed 's/^/    /' >>"${LOG_FILE}"

        # Local, light aggregation (under nice/ionice). Tiny pandas tabulation.
        if [ -f "${AGG_SCRIPT}" ]; then
            ${NICE} python3 "${AGG_SCRIPT}" >>"${LOG_FILE}" 2>&1 \
                && log "sweep: aggregate_diffusion_sweep.py ok -> ${SWEEP_SUMMARY}" \
                || log "sweep: WARN aggregate_diffusion_sweep.py returned non-zero"
        else
            log "sweep: WARN ${AGG_SCRIPT} not found — skipping aggregation"
        fi

        # Stage ONLY the new D_fit json + the regenerated summary. Never `git add -A`.
        git add -- "${DIFFUSION_DIR}/D_fit_"*.json 2>/dev/null
        [ -f "${SWEEP_SUMMARY}" ] && git add -- "${SWEEP_SUMMARY}" 2>/dev/null

        # Commit only if something is actually staged (idempotent).
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "data: babysit — landed sweep results" >>"${LOG_FILE}" 2>&1 \
                && log "sweep: committed" \
                || log "sweep: WARN git commit failed"

            # Push own repo only (origin = the user's own nife fork). Read-only on
            # everyone else: this pushes the user's branch to the user's remote.
            # No fetch/pull/stash anywhere — we never rebase onto upstream here.
            timeout "${SCP_TIMEOUT}" git push >>"${LOG_FILE}" 2>&1 \
                && log "sweep: pushed to origin" \
                || log "sweep: WARN git push failed/timed out (left committed locally)"
        else
            log "sweep: nothing staged after add — nothing to commit (idempotent)"
        fi
    else
        log "sweep: jobs idle, no new D_fit json — nothing to do"
    fi
else
    log "sweep: skipped (running=${RUNNING}; need a confirmed 0 to land results)"
fi

# ---------------------------------------------------------------------------
# (3) GPU RETRIEVE — read-only listing + scp of FINISHED files from vancouver01.
# ---------------------------------------------------------------------------
# Everything here is wrapped in `timeout`. If the host is down/slow we log and
# move on; the script still exits 0. We NEVER run remote compute — only `ls`.
mkdir -p "${LOCAL_GPU_DIR}" 2>/dev/null

# Read-only remote listing, newest-first. The `ls -t` only reads directory
# metadata; it does not touch or launch anything. Quote so globbing happens on
# the remote, not locally.
REMOTE_LIST=$(timeout "${SSH_TIMEOUT}" ssh ${SSH_OPTS} "${GPU_HOST}" \
    "ls -t ${GPU_RESULTS_DIR}/ 2>/dev/null" 2>/dev/null)
SSH_RC=$?

if [ "${SSH_RC}" -ne 0 ] || [ -z "${REMOTE_LIST}" ]; then
    if [ "${SSH_RC}" -eq 124 ]; then
        log "gpu: ssh ${GPU_HOST} TIMED OUT after ${SSH_TIMEOUT}s — skipping (exit 0)"
    elif [ "${SSH_RC}" -ne 0 ]; then
        log "gpu: ssh ${GPU_HOST} unreachable (rc=${SSH_RC}) — skipping (exit 0)"
    else
        log "gpu: ${GPU_HOST}:${GPU_RESULTS_DIR} empty/no results — nothing to retrieve"
    fi
else
    RETRIEVED=0
    # Iterate only result-like files (json/png). Pull a file iff it is missing
    # locally or strictly newer on the remote than our local copy.
    while IFS= read -r fname; do
        [ -z "${fname}" ] && continue
        case "${fname}" in
            *.json|*.png) : ;;     # keep
            *) continue ;;          # ignore logs, dirs, scratch
        esac

        local_path="${LOCAL_GPU_DIR}/${fname}"
        remote_path="${GPU_RESULTS_DIR}/${fname}"

        # Remote mtime (epoch). Read-only stat over ssh, timeout-guarded.
        rmtime=$(timeout "${SSH_TIMEOUT}" ssh ${SSH_OPTS} "${GPU_HOST}" \
            "stat -c %Y '${remote_path}' 2>/dev/null" 2>/dev/null)
        case "${rmtime}" in ''|*[!0-9]*) continue ;; esac   # unreadable -> skip

        # Local mtime (0 if absent => always fetch).
        if [ -f "${local_path}" ]; then
            lmtime=$(stat -c %Y "${local_path}" 2>/dev/null)
            case "${lmtime}" in ''|*[!0-9]*) lmtime=0 ;; esac
        else
            lmtime=0
        fi

        if [ "${rmtime}" -gt "${lmtime}" ]; then
            # scp the FINISHED file back, timeout-guarded, low priority.
            if timeout "${SCP_TIMEOUT}" ${NICE} scp ${SSH_OPTS} -p \
                    "${GPU_HOST}:${remote_path}" "${local_path}" >>"${LOG_FILE}" 2>&1; then
                log "gpu: retrieved ${fname} (remote newer)"
                RETRIEVED=$((RETRIEVED + 1))
            else
                log "gpu: WARN scp ${fname} failed/timed out — skipping"
            fi
        fi
    done <<EOF
${REMOTE_LIST}
EOF

    if [ "${RETRIEVED}" -gt 0 ]; then
        # Stage ONLY the retrieved-results dir. Never `git add -A`.
        git add -- "${LOCAL_GPU_DIR}/" 2>/dev/null
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "data: babysit — retrieved ${GPU_HOST} GPU results" >>"${LOG_FILE}" 2>&1 \
                && log "gpu: committed ${RETRIEVED} retrieved file(s)" \
                || log "gpu: WARN git commit failed"
            timeout "${SCP_TIMEOUT}" git push >>"${LOG_FILE}" 2>&1 \
                && log "gpu: pushed to origin" \
                || log "gpu: WARN git push failed/timed out (left committed locally)"
        else
            log "gpu: ${RETRIEVED} file(s) retrieved but no git change (already tracked, idempotent)"
        fi
    else
        log "gpu: nothing newer on ${GPU_HOST} — nothing retrieved"
    fi
fi

log "babysit: done"
exit 0
