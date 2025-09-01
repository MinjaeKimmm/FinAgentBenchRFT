import json, os, time, pathlib
from datetime import datetime
from openai import OpenAI

# Paths & constants
DATA_DIR     = "rft/data"
TRAIN_PATH   = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH     = os.path.join(DATA_DIR, "val.jsonl")
GRADER_SRC   = "rft/grader/ndcg_grader.py"        # must contain grade(sample,item)
MODEL_ID     = "o4-mini-2025-04-16"           # snapshot id required for RFT
N_EPOCHS     = 1
BATCH_SIZE   = 4
REASONING    = "medium"
EVAL_INTERVAL     = 10
EVAL_SAMPLES     = 1
SEED         = 42
COMPUTE_MULTIPLIER = 1 

OUTDIR = pathlib.Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=10000.0)

# Helper functions
def upload(path: str) -> str:
    print("Uploading", path, "…", end=" ", flush=True)
    fid = client.files.create(file=open(path, "rb"), purpose="fine-tune").id
    print(fid)
    return fid

def save_json(obj, fname: str):
    p = OUTDIR / fname
    p.write_text(json.dumps(obj, indent=2))
    return p

def format_time_remaining(estimated_finish: int, current_time: float) -> str:
    """Format estimated time remaining in human-readable format"""
    remaining = estimated_finish - current_time
    if remaining <= 0:
        return "overdue"
    
    hours = int(remaining // 3600)
    minutes = int((remaining % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def print_new_events(job_id: str, last_event_time: int) -> int:
    """Print new events since last_event_time, return newest event timestamp"""
    try:
        events = client.fine_tuning.jobs.list_events(job_id).data
        new_events = [e for e in events if e.created_at > last_event_time]
        
        if new_events:
            # Sort by timestamp (oldest first)
            new_events.sort(key=lambda x: x.created_at)
            
            for event in new_events:
                timestamp = datetime.fromtimestamp(event.created_at).strftime("%H:%M:%S")
                level = event.level.upper()
                message = event.message[:100] + "..." if len(event.message) > 100 else event.message
                print(f"  {timestamp} [{level:5}] {message}")
        
        return max(e.created_at for e in events) if events else last_event_time
    
    except Exception as e:
        print(f"  Error fetching events: {e.__class__.__name__}")
        return last_event_time

# Main
def main() -> None:
    # 1. upload datasets
    train_id = upload(TRAIN_PATH)
    val_id   = upload(VAL_PATH)

    # 2. read grader code
    with open(GRADER_SRC, encoding="utf-8") as f:
        grader_code = f.read()

    grader_obj = {
        "type": "python",
        "source": grader_code,
    }

    # 3. create fine-tune job
    job = client.fine_tuning.jobs.create(
        model=MODEL_ID,
        training_file=train_id,
        validation_file=val_id,
        method={
            "type": "reinforcement",
            "reinforcement": {
                "grader": grader_obj,
                "hyperparameters": {
                    "n_epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "compute_multiplier": COMPUTE_MULTIPLIER,
                    "reasoning_effort": REASONING,
                    "eval_interval": EVAL_INTERVAL,
                    "eval_samples": EVAL_SAMPLES
                },
            },
        },
        seed=SEED,
    )

    print("Job created:", job.id)
    print("Polling … (Ctrl-C to detach)")
    
    # Track last seen event timestamp
    last_event_time = job.created_at

    # 4. poll loop with exponential backoff
    retry_delay = 10  # start with 10 seconds
    max_retry_delay = 300  # cap at 5 minutes

    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job.id)
            current_time = time.time()
            
            # Format status line with time info
            status_line = f"{time.asctime()} status={job.status}"
            
            # Add estimated finish time if available
            if hasattr(job, 'estimated_finish') and job.estimated_finish:
                time_remaining = format_time_remaining(job.estimated_finish, current_time)
                status_line += f" | ETA: {time_remaining}"
            
            print(status_line)
            
            # Print new events
            last_event_time = print_new_events(job.id, last_event_time)
            
            if job.status in ("succeeded", "failed", "cancelled"):
                break
            retry_delay = 10  # reset delay on successful call
            time.sleep(10)

        except Exception as e:
            print(f"{time.asctime()} | Network error: {e.__class__.__name__}")
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, max_retry_delay)  # exponential backoff

    # 5. elapsed time (works even when finished_at absent)
    end_ts = getattr(job, "finished_at", None) or time.time()
    elapsed = end_ts - job.created_at
    print("Final status:", job.status, "| elapsed", f"{elapsed/60:.2f} min")

    # 6. dump job JSON
    job_path = save_json(job.model_dump(), f"job_{job.id}.json")
    print("Job JSON written to", job_path)

    # 7. fetch & save event log
    events = client.fine_tuning.jobs.list_events(job.id).data
    ev_path = save_json([ev.model_dump() for ev in events], f"events_{job.id}.json")
    print("Events JSON written to", ev_path)

    # 8. print only error / warning lines to console
    if job.status != "succeeded":
        print("\n--- validator / runtime messages ---")
        for ev in events:
            if ev.level in ("error", "warning"):
                ts = datetime.fromtimestamp(ev.created_at).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{ts} | {ev.level:7} | {ev.message}")

if __name__ == "__main__":
    main()