#!/bin/bash
# Nightly Claude Code analysis of pi-car-monitor
# Runs headless, analyses the day's logs, and fixes issues autonomously.
#
# Install: crontab -e → 0 3 * * * /home/PiAi/pi-car-monitor/scripts/nightly-analyse.sh
#
# Uses Max subscription (no API costs).

set -uo pipefail  # Don't exit on error (-e removed) — we want to handle failures

PROJECT_DIR="/home/PiAi/pi-car-monitor"
LOG_DIR="${PROJECT_DIR}/logs/nightly"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analysis_${TIMESTAMP}.log"
MAX_TURNS=30
TIMEOUT_SECONDS=900  # 15 minutes max

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

# Collect yesterday's stats for the prompt
ALERT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "Departure alert sent" || echo "0")
TIMEOUT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "departing_timeout" || echo "0")
IMPACT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "Recording started: impact" || echo "0")
VEHICLE_FP_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "ignoring vehicle" || echo "0")
FP_REPORTS=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "False positive reported" || echo "0")
ERRORS=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "\[ERROR\]" || echo "0")

log "Stats: departures=$ALERT_COUNT timeouts=$TIMEOUT_COUNT impacts=$IMPACT_COUNT filtered=$VEHICLE_FP_COUNT fp_reports=$FP_REPORTS errors=$ERRORS"

# Write prompt to temp file (avoids shell quoting issues)
PROMPT_FILE=$(mktemp)
cat > "$PROMPT_FILE" << 'PROMPT_END'
You are the overnight maintenance agent for pi-car-monitor. Read CLAUDE.md for full project context.

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

YOUR TASK:
1. Read the last 24h of journalctl logs: journalctl -u car-monitor.service --since "yesterday" --no-pager
2. Look for patterns: repeated FPs, errors, state issues, wasted user feedback
3. Read data/false_positives.yaml and data/transit_zones.yaml for learned zones
4. Read config/config.yaml for current thresholds
5. If you find clear fixes (threshold adjustments, bug fixes), implement them
6. Write a summary to this session's log file
7. Update CLAUDE.md changelog if you made code changes
8. Git add, commit with descriptive message, and push to origin/main

RULES:
- Make changes you're confident about. Document uncertainties but still fix obvious issues.
- Never change Telegram tokens, chat IDs, or security settings
- Never delete recordings or user data
- Keep changes minimal and focused
- The service will auto-reload on file changes, or the owner will restart it
- If something blocks you (permissions, missing files), log it clearly and continue with what you can do
- If this script itself has bugs, fix them
PROMPT_END

cd "$PROJECT_DIR"

log "Running Claude Code analysis..."

# Run Claude with dangerously-skip-permissions for full autonomy
# Using script wrapper for TTY, but Claude can now actually execute tools
timeout "$TIMEOUT_SECONDS" script -qec "claude -p \"\$(cat $PROMPT_FILE)\" \
    --dangerously-skip-permissions \
    --max-turns $MAX_TURNS \
    --output-format text" /dev/null >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    log "=== TIMEOUT: Analysis exceeded ${TIMEOUT_SECONDS}s ==="
elif [ $EXIT_CODE -ne 0 ]; then
    log "=== ERROR: Claude exited with code $EXIT_CODE ==="

    # If Claude failed, try to push whatever logs we have
    log "Attempting to commit error log for visibility..."
fi

log "=== Nightly analysis completed ==="

# Git operations — commit and push any changes
cd "$PROJECT_DIR"

# Check for changes in tracked files OR new files in safe directories
CHANGED=$(git diff --name-only HEAD 2>/dev/null || true)
UNTRACKED=$(git ls-files --others --exclude-standard -- src/ scripts/ CLAUDE.md 2>/dev/null || true)

if [ -n "$CHANGED" ] || [ -n "$UNTRACKED" ]; then
    log "Git: found changes to commit"

    # Stage safe files only
    git add src/ scripts/ CLAUDE.md 2>> "$LOG_FILE" || true
    git add config/config.yaml 2>> "$LOG_FILE" || true

    if ! git diff --cached --quiet 2>/dev/null; then
        DATE_STR=$(date +%Y-%m-%d)

        git commit -m "nightly: automated analysis ${DATE_STR}

Changes from overnight Claude Code maintenance agent.
See logs/nightly/analysis_${TIMESTAMP}.log for details.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" >> "$LOG_FILE" 2>&1

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
