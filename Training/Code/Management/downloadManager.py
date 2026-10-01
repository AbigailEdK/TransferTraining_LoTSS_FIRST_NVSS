#!/usr/bin/env python3
"""
Download Manager: Watch and restart checkDownloads.py if it freezes or crashes.

This script monitors the checkDownloads.py process and restarts it if:
- The process exits unexpectedly (crash)
- The process appears frozen (no new chunk output for a configurable timeout)

Similar in structure to Code/Management/manager.py but tailored for download monitoring.
Works on systems with or without tmux; runs the download in a subprocess.
"""
import subprocess
import os
import sys
from time import sleep, time
from datetime import datetime
import logging

# ===== Configuration =====
CHECKDOWNLOADS_SCRIPT = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Code/DataCapture/checkDownloads.py"
DATA_RESULTS_DIR = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Data/DATA_RESULTS"
REDOWNLOAD_CHUNKS_DIR = os.path.join(DATA_RESULTS_DIR, "redownload_chunks")
VENV_PYTHON = "/home/abigaildeklerk/Downloads/DeKlerk_Models/venv/bin/python"

# Monitoring parameters
CHECK_INTERVAL = 60  # seconds between health checks
FREEZE_TIMEOUT = 600  # seconds (10 minutes) of inactivity before declaring frozen
RESTART_DELAY = 10  # seconds to wait before restarting after crash

# ===== Logging setup =====
log_file = os.path.join(DATA_RESULTS_DIR, "downloadManager.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)


def get_latest_chunk_mtime():
    """Get the modification time of the latest chunk file, or None if none exist."""
    if not os.path.exists(REDOWNLOAD_CHUNKS_DIR):
        return None
    
    chunk_files = [f for f in os.listdir(REDOWNLOAD_CHUNKS_DIR) 
                   if f.startswith("chunk_") and f.endswith(".csv")]
    
    if not chunk_files:
        return None
    
    latest_chunk = max(chunk_files, 
                      key=lambda f: os.path.getmtime(os.path.join(REDOWNLOAD_CHUNKS_DIR, f)))
    return os.path.getmtime(os.path.join(REDOWNLOAD_CHUNKS_DIR, latest_chunk))


def is_process_running(proc):
    """Check if a subprocess is still running."""
    return proc is not None and proc.poll() is None


def start_checkdownloads():
    """Start checkDownloads.py as a subprocess with output streamed to console."""
    try:
        logging.info("Starting checkDownloads.py...")
        proc = subprocess.Popen(
            [VENV_PYTHON, CHECKDOWNLOADS_SCRIPT],
            stdout=None,  # inherit parent's stdout (show in console)
            stderr=None,  # inherit parent's stderr (show in console)
            text=True,
            bufsize=1
        )
        logging.info(f"checkDownloads.py started successfully (PID: {proc.pid})")
        return proc
    except Exception as e:
        logging.error(f"Failed to start checkDownloads.py: {e}")
        return None


def stop_process(proc):
    """Gracefully stop a subprocess, then forcefully kill if needed."""
    if proc is None:
        return
    
    try:
        if is_process_running(proc):
            logging.info(f"Terminating process (PID: {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logging.info("Process terminated gracefully")
            except subprocess.TimeoutExpired:
                logging.warning("Process did not terminate; forcing kill...")
                proc.kill()
                proc.wait()
                logging.info("Process killed")
    except Exception as e:
        logging.error(f"Error stopping process: {e}")


def get_chunk_progress():
    """Get the number of chunks completed and total missing indices."""
    if not os.path.exists(REDOWNLOAD_CHUNKS_DIR):
        return 0, "unknown"
    
    chunk_files = [f for f in os.listdir(REDOWNLOAD_CHUNKS_DIR) 
                   if f.startswith("chunk_") and f.endswith(".csv")]
    
    # Try to read scan_missing.csv to get total missing
    scan_file = os.path.join(REDOWNLOAD_CHUNKS_DIR, "scan_missing.csv")
    total_missing = "unknown"
    if os.path.exists(scan_file):
        try:
            import pandas as pd
            scan_df = pd.read_csv(scan_file)
            total_missing = len(scan_df[scan_df["missing_LOFAR"] | scan_df["missing_FIRST"] | scan_df["missing_NVSS"]])
        except Exception as e:
            logging.warning(f"Could not read scan_missing.csv: {e}")
    
    return len(chunk_files), total_missing


def main():
    logging.info("=" * 80)
    logging.info("Download Manager started")
    logging.info(f"Monitoring: {CHECKDOWNLOADS_SCRIPT}")
    logging.info(f"Results directory: {DATA_RESULTS_DIR}")
    logging.info(f"Health check interval: {CHECK_INTERVAL} seconds")
    logging.info(f"Freeze timeout: {FREEZE_TIMEOUT} seconds")
    logging.info("=" * 80)

    # Ensure results directory exists
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
    os.makedirs(REDOWNLOAD_CHUNKS_DIR, exist_ok=True)

    # Start checkDownloads.py
    proc = start_checkdownloads()
    if proc is None:
        logging.error("Failed to start checkDownloads.py on first attempt")
        sys.exit(1)

    last_chunk_mtime = get_latest_chunk_mtime()
    start_time = time()
    restart_count = 0

    try:
        while True:
            sleep(CHECK_INTERVAL)

            # Check if process is still running
            if not is_process_running(proc):
                logging.warning("checkDownloads.py process exited!")
                exit_code = proc.returncode
                logging.info(f"Exit code: {exit_code}")
                chunks, total = get_chunk_progress()
                logging.info(f"Progress before restart: {chunks} chunks completed, {total} total missing indices")
                
                # Restart it
                sleep(RESTART_DELAY)
                proc = start_checkdownloads()
                if proc is None:
                    logging.error("Failed to restart checkDownloads.py; will retry in next check")
                    proc = None
                    continue
                
                restart_count += 1
                last_chunk_mtime = get_latest_chunk_mtime()
                logging.info(f"Restart count: {restart_count}")
                continue

            # Check if process is frozen (no progress for FREEZE_TIMEOUT seconds)
            current_chunk_mtime = get_latest_chunk_mtime()
            
            if current_chunk_mtime is not None:
                time_since_last_chunk = time() - current_chunk_mtime
                
                if time_since_last_chunk > FREEZE_TIMEOUT:
                    logging.warning(f"No chunk update for {time_since_last_chunk:.0f} seconds (timeout: {FREEZE_TIMEOUT}s)")
                    logging.warning("Process appears frozen; restarting...")
                    chunks, total = get_chunk_progress()
                    logging.info(f"Progress before restart: {chunks} chunks completed, {total} total missing indices")
                    
                    stop_process(proc)
                    sleep(RESTART_DELAY)
                    proc = start_checkdownloads()
                    if proc is None:
                        logging.error("Failed to restart checkDownloads.py after freeze; will retry in next check")
                        continue
                    
                    restart_count += 1
                    last_chunk_mtime = get_latest_chunk_mtime()
                    logging.info(f"Restart count: {restart_count}")
                    continue
                else:
                    last_chunk_mtime = current_chunk_mtime
            
            # Status update
            elapsed = (time() - start_time) / 3600
            chunks, total = get_chunk_progress()
            time_since_chunk = time_since_last_chunk if current_chunk_mtime else "N/A"
            logging.info(f"[HEALTHY] PID: {proc.pid} | Elapsed: {elapsed:.2f}h | Restarts: {restart_count} | "
                        f"Progress: {chunks} chunks, {total} missing | Last update: {time_since_chunk if isinstance(time_since_chunk, str) else f'{time_since_chunk:.0f}s'} ago")

    except KeyboardInterrupt:
        logging.info("Manager interrupted by user")
        logging.info("Stopping checkDownloads.py...")
        stop_process(proc)
        logging.info("Shutdown complete")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Unexpected error in manager: {e}")
        stop_process(proc)
        sys.exit(1)


if __name__ == "__main__":
    main()
