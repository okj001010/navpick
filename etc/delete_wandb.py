import wandb
from datetime import datetime

api = wandb.Api()
runs = api.runs("okj001010/handreach")

cutoff = datetime.strptime("2025-07-25 13:00:00", "%Y-%m-%d %H:%M:%S")

for run in runs:
    name = run.name
    if not name.endswith("debug0"):
        continue
    try:
        date_str = name.split("_debug0")[0]  # '2025-07-25_13-25-56'
        dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        continue

    if dt < cutoff:
        print(f"Deleting run: {run.name} ({run.id})")
        run.delete()