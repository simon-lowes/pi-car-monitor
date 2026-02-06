#!/bin/bash
# Nightly Claude Code analysis of pi-car-monitor
# Runs headless, uses agent teams for cross-checking.
#
# Install: crontab -e → 0 3 * * * /home/PiAi/pi-car-monitor/scripts/nightly-analyse.sh
#
# Uses Max subscription (no API costs).

set -uo pipefail  # Don't exit on error (-e removed) — we want to handle failures

# Ensure claude is in PATH (cron doesn't load user profile)
export PATH="/home/PiAi/.local/bin:$PATH"
export HOME="/home/PiAi"

PROJECT_DIR="/home/PiAi/pi-car-monitor"
LOG_DIR="${PROJECT_DIR}/logs/nightly"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analysis_${TIMESTAMP}.log"
MAX_TURNS=75

mkdir -p "$LOG_DIR"

# Only run if we're not already running
LOCKFILE="/tmp/claude-nightly.lock"
if [ -f "$LOCKFILE" ]; then
    pid=$(cat "$LOCKFILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Already running (PID $pid), skipping" >> "$LOG_FILE"
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"

cleanup() {
    rm -f "$LOCKFILE" "$PROMPT_FILE" 2>/dev/null || true
}
trap cleanup EXIT

log() {
    echo "[$(date -Iseconds)] $1" >> "$LOG_FILE"
}

log "=== Nightly analysis started ==="

# Collect yesterday's stats — use subshell to avoid grep -c exit code 1 on zero matches
count_matches() {
    local count
    count=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "$1" 2>/dev/null) || true
    echo "${count:-0}"
}

ALERT_COUNT=$(count_matches "Departure alert sent")
TIMEOUT_COUNT=$(count_matches "departing_timeout")
IMPACT_COUNT=$(count_matches "Recording started: impact")
VEHICLE_FP_COUNT=$(count_matches "ignoring vehicle")
FP_REPORTS=$(count_matches "False positive reported")
ERRORS=$(count_matches "\[ERROR\]")

log "Stats: departures=$ALERT_COUNT timeouts=$TIMEOUT_COUNT impacts=$IMPACT_COUNT filtered=$VEHICLE_FP_COUNT fp_reports=$FP_REPORTS errors=$ERRORS"

# Write prompt to temp file (avoids shell quoting issues)
PROMPT_FILE=$(mktemp)
cat > "$PROMPT_FILE" << 'PROMPT_END'
You are the overnight maintenance lead for pi-car-monitor. Read CLAUDE.md for full project context.

YESTERDAY'S STATS:
PROMPT_END

cat >> "$PROMPT_FILE" << STATS_END
- Departure alerts sent: ${ALERT_COUNT}
- Departing timeouts (false departures): ${TIMEOUT_COUNT}
- Impact recordings: ${IMPACT_COUNT}
- Vehicles filtered (depth/passing/transit): ${VEHICLE_FP_COUNT}
- User FP reports (null replies): ${FP_REPORTS}
- Errors logged: ${ERRORS}
STATS_END

cat >> "$PROMPT_FILE" << 'PROMPT_END'

YOUR TASK — USE AGENT TEAMS:
You MUST use agent teams (TeamCreate) to coordinate multiple agents that cross-check each other's work. Here is your workflow:

1. Create a team with TeamCreate (e.g. team_name "nightly-analysis")
2. Create tasks with TaskCreate for each phase of work
3. Spawn the following agents as teammates using the Task tool with team_name:

   AGENT 1 — "log-analyst" (subagent_type: general-purpose)
   Task: Read the last 24h of journalctl logs (journalctl -u car-monitor.service --since "yesterday" --no-pager), read data/false_positives.yaml, data/transit_zones.yaml, and config/config.yaml. Identify patterns: repeated FPs, errors, state machine issues, wasted feedback, threshold problems. Report findings with specific evidence (log lines, timestamps, counts). Propose specific fixes with file paths and what to change.

   AGENT 2 — "code-reviewer" (subagent_type: general-purpose)
   Task: Wait for the log-analyst's findings. Then review each proposed fix by reading the relevant source files. For each proposal, assess: (a) Is the diagnosis correct? (b) Will the fix work? (c) Could it cause regressions? (d) Is there a simpler approach? Reject proposals that are risky or poorly justified. Approve proposals that are safe and well-evidenced. Be skeptical — the analyst might misread logs or propose over-engineered fixes.

4. As team lead, you coordinate:
   - First assign the analyst task, wait for their report
   - Then assign the reviewer task with the analyst's findings
   - Only implement fixes that the reviewer APPROVES
   - If they disagree, use your judgement but err on the side of caution
   - Implement approved fixes yourself (you are the only one who edits code)

5. After implementing:
   - Update CLAUDE.md changelog if you made code changes
   - Git add, commit with descriptive message, and push via: gh auth setup-git && git push origin main
   - Shut down teammates when done (SendMessage with type: shutdown_request)
   - Delete the team with TeamDelete

6. If no fixes are needed, just log "No issues found" and clean up.

RULES:
- Only implement fixes that survive scrutiny from the reviewer agent
- Never change Telegram tokens, chat IDs, or security settings
- Never delete recordings or user data
- Keep changes minimal and focused
- The service needs manual restart to apply code changes — note this in the commit message
- If something blocks you, log it clearly and continue with what you can do
- If this script itself has bugs, fix them
PROMPT_END

cd "$PROJECT_DIR"

log "Running Claude Code analysis with agent teams..."

# Run Claude with dangerously-skip-permissions for full autonomy
# No timeout — let the agent teams run to completion. The lockfile
# prevents overlapping runs. If it hangs, the next night's run
# will be skipped and the owner can investigate.
script -qec "claude -p \"\$(cat $PROMPT_FILE)\" \
    --dangerously-skip-permissions \
    --max-turns $MAX_TURNS \
    --output-format text" /dev/null >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    log "=== ERROR: Claude exited with code $EXIT_CODE ==="
    log "Attempting to commit error log for visibility..."
fi

log "=== Nightly analysis completed ==="

# Git operations — commit and push any changes Claude didn't already commit
cd "$PROJECT_DIR"

# Check for changes in tracked files OR new files in safe directories
CHANGED=$(git diff --name-only HEAD 2>/dev/null || true)
UNTRACKED=$(git ls-files --others --exclude-standard -- src/ scripts/ CLAUDE.md 2>/dev/null || true)

if [ -n "$CHANGED" ] || [ -n "$UNTRACKED" ]; then
    log "Git: found uncommitted changes, committing..."

    # Stage safe files only
    git add src/ scripts/ CLAUDE.md 2>> "$LOG_FILE" || true
    git add config/config.yaml 2>> "$LOG_FILE" || true

    if ! git diff --cached --quiet 2>/dev/null; then
        DATE_STR=$(date +%Y-%m-%d)

        git commit -m "nightly: automated analysis ${DATE_STR}

Changes from overnight Claude Code maintenance agent team.
See logs/nightly/analysis_${TIMESTAMP}.log for details.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" >> "$LOG_FILE" 2>&1

        # Push
        gh auth setup-git >> "$LOG_FILE" 2>&1 || true
        if git push origin main >> "$LOG_FILE" 2>&1; then
            log "Git: pushed to origin/main"
        else
            log "Git: push failed (will retry next night)"
        fi
    else
        log "Git: nothing staged after filtering"
    fi
else
    log "Git: no changes detected"
fi

# Prune old logs (keep last 30 days)
find "$LOG_DIR" -name "analysis_*.log" -mtime +30 -delete 2>/dev/null || true

log "=== Script finished ==="
