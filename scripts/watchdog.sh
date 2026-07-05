#!/usr/bin/env bash
# watchdog.sh — overnight LightRAG monitor
# Runs every 5 minutes. If LightRAG is down or pipeline stalled, restarts it.
# After both books complete, uploads journal_articles/ in batches of 100,
# waiting for each batch to fully complete before sending the next.
#
# Usage: bash scripts/watchdog.sh >> /tmp/watchdog.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIGHTRAG_DIR="$REPO_ROOT/LightRAG"
JOURNAL_DIR="$REPO_ROOT/journal_articles"
LOG="/tmp/lightrag.log"
SERVER_URL="http://localhost:9621"
CHECK_INTERVAL=300   # 5 minutes
STALL_THRESHOLD=7200 # restart if no activity for 2 hours while busy
                     # HNSW writes to remote Supabase are silent for 30-60+ min

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── helpers ──────────────────────────────────────────────────────────────────

server_up() {
    curl -s --max-time 5 "$SERVER_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='healthy' else 1)" 2>/dev/null
}

pipeline_busy() {
    curl -s --max-time 5 "$SERVER_URL/documents/pipeline_status" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('busy','false'))" 2>/dev/null
}

doc_counts() {
    # prints: completed=N failed=N processing=N pending=N
    for status in completed failed processing pending; do
        n=$(curl -s --max-time 5 "$SERVER_URL/documents?status=$status&page=1&page_size=1" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('statuses',{}); print(len(s.get('$status',[])))" 2>/dev/null)
        printf "%s=%s " "$status" "${n:-?}"
    done
    echo
}

PROXY_SCRIPT="$REPO_ROOT/scripts/openai_proxy.py"
PROXY_PYTHON="$REPO_ROOT/LightRAG/.venv/bin/python3"

ensure_proxy() {
    if ! curl -s --max-time 3 "http://localhost:8080/health" | grep -q '"status"'; then
        log "Proxy down — restarting..."
        pkill -f "openai_proxy.py" 2>/dev/null
        nohup "$PROXY_PYTHON" "$PROXY_SCRIPT" >> /tmp/proxy.log 2>&1 &
        sleep 3
        log "Proxy restarted (PID $!)"
    fi
}

start_server() {
    log "Starting LightRAG server..."
    ensure_proxy
    cd "$LIGHTRAG_DIR" || return 1
    nohup .venv/bin/lightrag-server >> "$LOG" 2>&1 &
    local pid=$!
    log "Server started with PID $pid, waiting 90s for startup..."
    sleep 90
    if server_up; then
        log "Server is healthy."
        return 0
    else
        log "ERROR: Server did not come up after 90s."
        return 1
    fi
}

trigger_reprocess() {
    log "Triggering reprocess_failed..."
    curl -s -X POST "$SERVER_URL/documents/reprocess_failed" \
        -H "Content-Type: application/json" | python3 -m json.tool 2>/dev/null || true
}

# ── journal articles bulk upload (batched) ────────────────────────────────────

BATCH_SIZE=100  # upload this many PDFs, then wait for pipeline to finish

upload_batch() {
    # upload_batch PDF1 PDF2 ... — uploads a slice of PDFs and returns
    local uploaded=0
    for pdf in "$@"; do
        local name
        name=$(basename "$pdf")
        local response
        response=$(curl -s --max-time 60 -X POST "$SERVER_URL/documents/upload" \
            -F "file=@$pdf" 2>&1)
        if echo "$response" | grep -q '"status"'; then
            log "  Uploaded: $name"
            ((uploaded++))
        else
            log "  WARN upload failed for $name: $(echo "$response" | head -c 200)"
        fi
        sleep 1  # brief pause between uploads to avoid overwhelming the server
    done
    log "  Batch: $uploaded / $# files uploaded successfully."
}

wait_for_pipeline() {
    # Wait until pipeline is idle (not busy) — poll every 2 minutes
    log "  Waiting for pipeline to finish batch..."
    sleep 30  # give it a moment to start
    local wait_count=0
    while true; do
        local busy
        busy=$(pipeline_busy)
        if [[ "$busy" == "False" ]]; then
            log "  Pipeline idle — batch complete."
            return 0
        fi
        # Also check for failed docs and reprocess them while waiting
        local failed
        failed=$(curl -s --max-time 5 "$SERVER_URL/documents?status=failed&page=1&page_size=1" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('statuses',{}); print(len(s.get('failed',[])))" 2>/dev/null)
        if [[ "${failed:-0}" -gt 0 ]]; then
            log "  Found $failed failed doc(s) mid-batch — triggering reprocess..."
            trigger_reprocess
        fi
        (( wait_count++ ))
        if (( wait_count % 6 == 0 )); then
            log "  Still processing... (waited $(( wait_count * 2 )) min)"
        fi
        sleep 120  # check every 2 minutes
    done
}

upload_journals() {
    local all_pdfs=("$JOURNAL_DIR"/*.pdf)
    if [[ ${#all_pdfs[@]} -eq 0 ]] || [[ ! -e "${all_pdfs[0]}" ]]; then
        log "No PDFs found in $JOURNAL_DIR — skipping journal upload."
        return
    fi
    local total=${#all_pdfs[@]}
    log "Found $total PDFs in journal_articles/. Uploading in batches of $BATCH_SIZE..."

    local batch_num=0
    local i=0
    while (( i < total )); do
        (( batch_num++ ))
        local end=$(( i + BATCH_SIZE ))
        (( end > total )) && end=$total
        local batch=("${all_pdfs[@]:$i:$BATCH_SIZE}")
        log "--- Batch $batch_num: files $((i+1))–$end of $total ---"
        upload_batch "${batch[@]}"
        wait_for_pipeline
        i=$end
    done
    log "=== All $total journal PDFs uploaded and processed. ==="
}

# ── main loop ─────────────────────────────────────────────────────────────────

LAST_LOG_SIZE=0
LAST_ACTIVITY_TIME=$(date +%s)
LAST_PIPELINE_MSG=""
JOURNALS_UPLOADED=false

log "=== Watchdog started. Monitoring LightRAG. Check interval: ${CHECK_INTERVAL}s ==="

while true; do
    # 0. Ensure proxy is up (LightRAG needs it for all LLM calls)
    ensure_proxy

    # 1. Ensure server is up
    if ! server_up; then
        log "Server is DOWN — restarting..."
        pkill -f "lightrag-server" 2>/dev/null
        sleep 3
        if start_server; then
            trigger_reprocess
            LAST_LOG_SIZE=$(wc -l < "$LOG")
            LAST_ACTIVITY_TIME=$(date +%s)
        fi
    fi

    # 2. Check for stall (busy but no activity for STALL_THRESHOLD seconds)
    # Activity = log file growing OR pipeline latest_message changing.
    # DB writes during merge are silent (no log lines) but the message changes.
    CURRENT_LOG_SIZE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    CURRENT_PIPELINE_MSG=$(curl -s --max-time 5 "$SERVER_URL/documents/pipeline_status" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('latest_message',''))" 2>/dev/null)
    BUSY=$(pipeline_busy)
    NOW=$(date +%s)

    if [[ "$CURRENT_LOG_SIZE" -gt "$LAST_LOG_SIZE" ]] || \
       [[ -n "$CURRENT_PIPELINE_MSG" && "$CURRENT_PIPELINE_MSG" != "$LAST_PIPELINE_MSG" ]]; then
        LAST_ACTIVITY_TIME=$NOW
        LAST_LOG_SIZE=$CURRENT_LOG_SIZE
        LAST_PIPELINE_MSG="$CURRENT_PIPELINE_MSG"
    fi

    IDLE_SECS=$(( NOW - LAST_ACTIVITY_TIME ))

    if [[ "$BUSY" == "True" ]] && [[ "$IDLE_SECS" -gt "$STALL_THRESHOLD" ]]; then
        # Only restart if server is ALSO unresponsive.
        # A healthy+busy server that's silent is just doing a long HNSW write —
        # restarting it would cancel the in-progress executemany and waste hours.
        if ! server_up; then
            log "Pipeline stalled AND server unresponsive (idle ${IDLE_SECS}s) — restarting..."
            pkill -f "lightrag-server" 2>/dev/null
            sleep 3
            if start_server; then
                trigger_reprocess
                LAST_LOG_SIZE=$(wc -l < "$LOG")
                LAST_ACTIVITY_TIME=$(date +%s)
            fi
        else
            log "Pipeline silent for ${IDLE_SECS}s but server is healthy — long DB write in progress, not restarting."
            LAST_ACTIVITY_TIME=$(date +%s)  # reset so this logs once per threshold, not every 5 min
        fi

    elif [[ "$BUSY" == "False" ]] && server_up; then
        # 3. Check if there are failed docs to reprocess
        FAILED=$(curl -s --max-time 5 "$SERVER_URL/documents" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('statuses',{}); print(len(s.get('failed',[])))" 2>/dev/null)
        if [[ "$FAILED" -gt 0 ]]; then
            log "Found $FAILED failed document(s) — triggering reprocess..."
            trigger_reprocess
            LAST_ACTIVITY_TIME=$(date +%s)

        elif [[ "$JOURNALS_UPLOADED" == "false" ]]; then
            # Check if both original books are processed (LightRAG uses "processed" not "completed")
            COMPLETED=$(curl -s --max-time 5 "$SERVER_URL/documents" \
                | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('statuses',{}); print(len(s.get('processed',[])))" 2>/dev/null)
            if [[ "${COMPLETED:-0}" -ge 2 ]]; then
                log "Both books processed (processed=$COMPLETED). Starting journal article upload..."
                upload_journals
                JOURNALS_UPLOADED=true
            else
                log "Status: completed=$COMPLETED, pipeline idle. Waiting..."
            fi
        else
            log "Status: pipeline idle, journals uploaded. All done."
            COUNTS=$(doc_counts)
            log "Document counts: $COUNTS"
        fi
    else
        log "Status: busy=$BUSY, log_lines=$CURRENT_LOG_SIZE, idle_secs=$IDLE_SECS"
    fi

    sleep "$CHECK_INTERVAL"
done
