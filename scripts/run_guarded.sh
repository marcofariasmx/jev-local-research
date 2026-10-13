#!/bin/zsh
# Runs the DMR baseline under a memory watchdog; always unloads the model at the end.
cd "$(dirname "$0")/.."
MODEL=${MODEL:-ai/qwen3.5:4b-iq4_XS}; OUT=${OUT:-dmr4b-authored144.jsonl}; RSS_MAX_MB=${RSS_MAX_MB:-6656}
docker model configure --context-size 8192 $MODEL -- --cache-ram 0 >/dev/null 2>&1
swap0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
semif/.venv/bin/python scripts/dmr_baseline.py $MODEL semif/benchmarks/data/authored144.jsonl results/raw/$OUT > dmr.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -axo rss,comm | awk '/llama-server/{s+=$1} END{print int(s/1024)}')
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  if (( rss > $RSS_MAX_MB || swap - swap0 > 3500 )); then
    echo "WATCHDOG: llama-server ${rss} MB, swap +$((swap - swap0)) MB -> abort" >> dmr.log
    kill $pid; break
  fi
  sleep 1
done
wait $pid; echo "EXIT $?" >> dmr.log
docker model unload --all >> dmr.log 2>&1
