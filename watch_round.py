"""Poll the event clock and submit roster.txt to round 1 the moment it opens, then exit."""
import os, subprocess, sys, time
from datetime import datetime, timezone
from everestapi import EverestAPI

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], base_url="https://hackathon.everesteer.ai")
TARGET_ROUND = sys.argv[1] if len(sys.argv) > 1 else "round_1"

while True:
    try:
        cad = client.get_started().get("cadence") or {}
    except Exception as e:  # transient API errors: keep polling
        print("poll error", e, flush=True); time.sleep(30); continue
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    ready = cad.get("open_window") and not cad.get("intake_fenced") and cad.get("live_data_available")
    if ready and TARGET_ROUND in str(cad.get("open_window")):
        roster = open("roster.txt").read().split()
        print(now, "ROUND OPEN:", cad["open_window"], "submitting", roster, flush=True)
        for attempt in range(5):
            r = subprocess.run([sys.executable, "round.py", *roster], capture_output=True, text=True)
            print(r.stdout, r.stderr[-2000:], flush=True)
            if r.returncode == 0:
                sys.exit(0)
            time.sleep(20)
        sys.exit("SUBMIT FAILED after 5 attempts")
    print(now, cad.get("phase"), "next phase in", cad.get("seconds_until_next_phase"), "s", flush=True)
    time.sleep(30)
