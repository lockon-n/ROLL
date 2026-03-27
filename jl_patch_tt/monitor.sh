#!/bin/bash
# Monitor CPU and memory usage for the current user's processes
# Usage: bash monitor.sh [interval_seconds] (default: 2)

INTERVAL=${1:-2}
USER=$(whoami)

tput civis 2>/dev/null
trap 'tput cnorm 2>/dev/null; exit' INT TERM

clear

while true; do
    # Cursor to top-left
    tput cup 0 0

    # Header
    tput el; echo "========== $(date '+%Y-%m-%d %H:%M:%S') | User: $USER | Interval: ${INTERVAL}s =========="
    tput el; echo ""

    # System overview
    tput el; echo "--- System Overview ---"
    top -bn1 | head -4 | tail -2 | while IFS= read -r line; do
        tput el; echo "$line"
    done
    tput el; echo ""

    # User total
    read TOTAL_CPU TOTAL_RSS TOTAL_VSZ PROC_COUNT <<< $(ps -u "$USER" -o pcpu,rss,vsz --no-headers 2>/dev/null | \
        awk '{cpu+=$1; rss+=$2; vsz+=$3; n++} END {printf "%.1f %.0f %.0f %d", cpu, rss, vsz, n}')
    tput el; echo "--- User Total (${PROC_COUNT} processes) ---"
    CORES=$(echo "scale=1; $TOTAL_CPU / 100" | bc -l)
    RSS_GB=$(echo "scale=1; $TOTAL_RSS / 1048576" | bc -l)
    VSZ_GB=$(echo "scale=1; $TOTAL_VSZ / 1048576" | bc -l)
    tput el; echo "  CPU: ${TOTAL_CPU}% (~${CORES} cores)    RSS: ${RSS_GB} GB    VSZ: ${VSZ_GB} GB"
    tput el; echo ""

    # Top processes
    tput el; echo "--- Top Processes by CPU ---"
    tput el; echo "PID         CPU%   RSS(GB)  COMMAND"
    tput el; echo "-----------------------------------------------"
    ps -u "$USER" -o pid,pcpu,rss,comm --no-headers --sort=-pcpu 2>/dev/null | head -15 | \
        while read PID CPU RSS COMM; do
            RSS_GB=$(echo "scale=2; $RSS / 1048576" | bc -l)
            tput el
            echo "$PID  $CPU  $RSS_GB  $COMM"
        done
    tput el; echo ""

    # GPU usage
    if command -v nvidia-smi &>/dev/null; then
        tput el; echo "--- GPU Usage ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            while IFS=', ' read -r IDX UTIL MEM_USED MEM_TOTAL; do
                tput el; echo "  GPU $IDX: Util ${UTIL}%  Memory ${MEM_USED}/${MEM_TOTAL} MiB"
            done
    fi

    # Clear any leftover lines from previous longer output
    tput ed

    sleep "$INTERVAL"
done
