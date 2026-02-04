#!/bin/bash
# Nightly Claude Code analysis of pi-car-monitor
# Runs headless, analyses the day's logs, and fixes issues autonomously.
#
# Install: crontab -e → 0 3 * * * /home/PiAi/pi-car-monitor/scripts/nightly-analyse.sh
#
# Uses Max subscription (no API costs). The script -qc wrapper is required
# because claude -p has a TTY bug that causes it to hang without a pseudo-TTY.

set -euo pipefail

PROJECT_DIR="/home/PiAi/pi-car-monitor"
LOG_DIR="${PROJECT_DIR}/logs/nightly"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analysis_${TIMESTAMP}.log"
MAX_TURNS=25
TIMEOUT_SECONDS=600  # 10 minutes max

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
trap "rm -f $LOCKFILE" EXIT

echo "=== Nightly analysis started at $(date -Iseconds) ===" >> "$LOG_FILE"

# Collect today's stats for the prompt
ALERT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "Departure alert sent" || echo "0")
TIMEOUT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "departing_timeout" || echo "0")
IMPACT_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "Recording started: impact" || echo "0")
VEHICLE_FP_COUNT=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "ignoring vehicle" || echo "0")
FP_REPORTS=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "False positive reported" || echo "0")
ERRORS=$(journalctl -u car-monitor.service --since "yesterday" --until "today" --no-pager 2>/dev/null | grep -c "\[ERROR\]" || echo "0")

PROMPT="You are the overnight maintenance agent for pi-car-monitor. Your CLAUDE.md has full project context.

YESTERDAY'S STATS:
- Departure alerts sent: ${ALERT_COUNT}
- Departing timeouts (false departures): ${TIMEOUT_COUNT}
- Impact recordings: ${IMPACT_COUNT}
- Vehicles filtered (depth/passing/transit): ${VEHICLE_FP_COUNT}
- User FP reports (null replies): ${FP_REPORTS}
- Errors logged: ${ERRORS}

YOUR TASK:
1. Read the last 24h of journalctl logs for car-monitor.service. Focus on patterns — repeated false positives, errors, state machine issues.
2. Read data/false_positives.yaml and data/transit_zones.yaml for learned zones.
3. Read the current detection thresholds in config/config.yaml.
4. Identify any systematic false positive patterns that code changes could fix.
5. If you find clear, safe fixes (threshold adjustments, new filters for proven FP patterns), implement them.
6. DO NOT restart the service — the owner will do that.
7. Write a brief summary of findings and any changes to logs/nightly/analysis_${TIMESTAMP}.log.
8. Update the project CLAUDE.md changelog if you made code changes.

RULES:
- Only make changes you are confident about. If unsure, document the issue but don't change code.
- Never change Telegram bot tokens, chat IDs, or security settings.
- Never delete recordings or user data.
- Keep changes minimal and focused. One clear fix per issue.
- If no issues found, just write 'No issues detected' to the log and exit."

cd "$PROJECT_DIR"

# Write prompt to a temp file to avoid shell quoting issues
# (printf %q produces Bash-specific escapes that sh can't parse)
PROMPT_FILE=$(mktemp)
echo "$PROMPT" > "$PROMPT_FILE"
trap "rm -f $LOCKFILE $PROMPT_FILE" EXIT

timeout "$TIMEOUT_SECONDS" script -qc "claude -p \"\$(cat $PROMPT_FILE)\" \
    --allowedTools 'Bash,Read,Edit,Write,Grep,Glob' \
    --max-turns $MAX_TURNS \
    --output-format text" /dev/null >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "=== Analysis timed out after ${TIMEOUT_SECONDS}s ===" >> "$LOG_FILE"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "=== Analysis exited with code $EXIT_CODE ===" >> "$LOG_FILE"
fi

echo "=== Nightly analysis completed at $(date -Iseconds) ===" >> "$LOG_FILE"

# Git commit and push any changes made by the agent
cd "$PROJECT_DIR"
CHANGED_FILES=$(git diff --name-only HEAD 2>/dev/null || true)
UNTRACKED_FILES=$(git ls-files --others --exclude-standard -- src/ config/ CLAUDE.md scripts/ 2>/dev/null || true)

if [ -n "$CHANGED_FILES" ] || [ -n "$UNTRACKED_FILES" ]; then
    echo "=== Git: committing changes ===" >> "$LOG_FILE"

    # Stage only safe files (source, config, docs, scripts — never data/recordings/secrets)
    git add src/ config/config.yaml CLAUDE.md scripts/ 2>> "$LOG_FILE" || true

    # Check if there's actually anything staged
    if ! git diff --cached --quiet 2>/dev/null; then
        DATE_STR=$(date +%Y-%m-%d)
        git commit -m "$(cat <<EOF
nightly: automated analysis ${DATE_STR}

Changes from overnight Claude Code maintenance agent.
See logs/nightly/analysis_${TIMESTAMP}.log for details.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)" >> "$LOG_FILE" 2>&1

        # Push using gh-authenticated git
        gh auth setup-git >> "$LOG_FILE" 2>&1
        git push origin main >> "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "=== Git: pushed to origin/main ===" >> "$LOG_FILE"
        else
            echo "=== Git: push failed (will retry next night) ===" >> "$LOG_FILE"
        fi
    else
        echo "=== Git: no staged changes to commit ===" >> "$LOG_FILE"
    fi
else
    echo "=== Git: no changes detected ===" >> "$LOG_FILE"
fi

# Prune old logs (keep last 30 days)
find "$LOG_DIR" -name "analysis_*.log" -mtime +30 -delete 2>/dev/null || true
