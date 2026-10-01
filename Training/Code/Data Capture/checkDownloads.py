#!/usr/bin/env python3
"""Check downloaded FITS files and attempt to redownload missing ones in resumable chunks.

This script:
- Iterates through all indices in the catalogue (same as download_images.py)
- Checks if the three FITS files (LOFAR, FIRST, NVSS) exist for each index
- For missing files, attempts redownload up to 2 additional times (3 total attempts)
- Processes indices in chunks of 100 (configurable)
- Saves chunk progress so runs can be resumed after interruptions
- Can be rerun repeatedly to retry failed downloads until satisfied

Chunk files are saved in a subdirectory and can be inspected/deleted if needed.
"""
import argparse
import os
import time
import logging
import pandas as pd
import glob

# Import downloader helpers from the module in the same directory
from download_images import get_lofar, get_first, get_nvss, download, ping_lofar


def expected_paths(output_dir, index, morph, cat, ra, dec):
    """Return expected file paths for the three surveys for a source row.
    Uses the same filename pattern as `download_images.download()`.
    """
    filename_pre = f"{output_dir}/{index}_"
    filename_post = f"_{morph}_{cat}_{ra}_{dec}.fits"
    return {
        "LOFAR": filename_pre + "LOFAR" + filename_post,
        "FIRST": filename_pre + "FIRST" + filename_post,
        "NVSS": filename_pre + "NVSS" + filename_post,
    }


def try_redownload(index, entry, output_dir, fov_deg=0.125, match_shape=True, max_attempts=2):
    """Try to redownload missing files for a single catalogue entry.
    
    Args:
        index: catalogue index
        entry: row from DataFrame with RA/deg, DEC/deg, Type, Catalog
        output_dir: where to save FITS files
        fov_deg: field of view in degrees
        match_shape: whether to use LOFAR shape for FIRST/NVSS
        max_attempts: number of attempts (default 2 = 3 total tries)
    
    Returns a dict with keys: attempted, success, missing_before, missing_after, messages
    """
    ra = entry["RA/deg"]
    dec = entry["DEC/deg"]
    morph = entry["Type"]
    cat = entry["Catalog"]

    pats = expected_paths(output_dir, index, morph, cat, ra, dec)
    missing_before = {k: not os.path.exists(p) for k, p in pats.items()}

    result = {
        "attempted": any(missing_before.values()),
        "success": False,
        "missing_before": missing_before,
        "missing_after": dict(missing_before),
        "messages": [],
    }

    # If all files already exist, no attempt needed
    if not result["attempted"]:
        result["success"] = True
        result["messages"].append("all present")
        return result

    # Try to (re)download up to max_attempts times
    for attempt in range(1, max_attempts + 1):
        try:
            # LOFAR: returns (response, shape) or (None, None)
            response_lofar, shape_lofar = get_lofar(index, ra, dec, fov_deg)
            if response_lofar is None:
                msg = f"LOFAR attempt {attempt} returned no data"
                logging.info(msg)
                result["messages"].append(msg)
            
            # Determine pixels for FIRST/NVSS based on LOFAR shape
            pixels = shape_lofar if (shape_lofar is not None and match_shape) else None

            first_hdu = get_first(index, ra, dec, fov_deg, pixels)
            if first_hdu is None:
                msg = f"FIRST attempt {attempt} returned no data"
                logging.info(msg)
                result["messages"].append(msg)

            nvss_hdu = get_nvss(index, ra, dec, fov_deg, pixels)
            if nvss_hdu is None:
                msg = f"NVSS attempt {attempt} returned no data"
                logging.info(msg)
                result["messages"].append(msg)

            # Only write files if we have all three
            if response_lofar is not None and first_hdu is not None and nvss_hdu is not None:
                download(response_lofar, first_hdu, nvss_hdu, index, ra, dec, morph, cat, output_dir)
                
                # Check files now exist
                pats2 = expected_paths(output_dir, index, morph, cat, ra, dec)
                missing_after = {k: not os.path.exists(p) for k, p in pats2.items()}
                result["missing_after"] = missing_after
                
                if not any(missing_after.values()):
                    result["success"] = True
                    result["messages"].append(f"downloaded on attempt {attempt}")
                    break
                else:
                    result["messages"].append(f"files still missing after write on attempt {attempt}")
            else:
                # If any surveys failed, note it
                if response_lofar is None:
                    result["messages"].append(f"LOFAR not available on attempt {attempt}")
                if first_hdu is None:
                    result["messages"].append(f"FIRST not available on attempt {attempt}")
                if nvss_hdu is None:
                    result["messages"].append(f"NVSS not available on attempt {attempt}")

        except Exception as e:
            logging.exception(f"Exception while trying to redownload {index} on attempt {attempt}: {e}")
            result["messages"].append(f"exception attempt {attempt}: {str(e)[:50]}")

        # Brief sleep between attempts
        if attempt < max_attempts:
            time.sleep(1)

    # Final check: refresh missing_after if not already successful
    if not result["success"]:
        pats2 = expected_paths(output_dir, index, morph, cat, ra, dec)
        result["missing_after"] = {k: not os.path.exists(p) for k, p in pats2.items()}
        if any(result["missing_after"].values()):
            result["messages"].append("files still missing after all attempts")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check downloaded FITS files and attempt to redownload missing ones in resumable chunks.",
        epilog="Chunk files are saved in output_dir/redownload_chunks/. You can rerun this script to retry failed downloads."
    )
    parser.add_argument("--df", 
        default="/home/abigaildeklerk/Downloads/DeKlerk_Models/Code/DataCapture/RADCAT_nonunique.csv", 
        help="catalogue CSV used for downloads")
    parser.add_argument("--output-dir", 
        default="/home/abigaildeklerk/Downloads/DeKlerk_Models/Data/DATA_RESULTS", 
        help="output directory where FITS are stored")
    parser.add_argument("--fov-deg", type=float, default=0.125, 
        help="field of view used for downloads (degrees)")
    parser.add_argument("--chunk-size", type=int, default=100, 
        help="number of indices to process per chunk (default 100)")
    parser.add_argument("--max-attempts", type=int, default=2, 
        help="number of retry attempts per missing source (default 2 = 3 total tries)")
    parser.add_argument("--scan-only", action="store_true", 
        help="only scan for missing files; don't attempt downloads")
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load catalogue
    df = pd.read_csv(args.df, index_col=0)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Create chunks subdirectory for progress tracking
    chunks_dir = os.path.join(out_dir, "redownload_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Ping LOFAR once before starting
    try:
        ping_lofar()
    except Exception as e:
        logging.warning(f"ping_lofar failed: {e}")

    # ===== Phase 1: Scan for missing files =====
    logging.info(f"Scanning {len(df)} catalogue entries for missing files...")
    missing_indices = []
    scan_rows = []
    
    for idx, (index, entry) in enumerate(df.iterrows()):
        if (idx + 1) % 1000 == 0:
            logging.info(f"Scanned {idx + 1}/{len(df)} entries...")
        
        pats = expected_paths(out_dir, index, entry["Type"], entry["Catalog"], 
                             entry["RA/deg"], entry["DEC/deg"])
        exists = {k: os.path.exists(p) for k, p in pats.items()}
        missing = any(not v for v in exists.values())
        
        scan_rows.append({
            "index": index,
            "missing_LOFAR": not exists["LOFAR"],
            "missing_FIRST": not exists["FIRST"],
            "missing_NVSS": not exists["NVSS"],
        })
        
        if missing:
            missing_indices.append(index)

    # Save scan results
    scan_df = pd.DataFrame(scan_rows)
    scan_path = os.path.join(chunks_dir, "scan_missing.csv")
    scan_df.to_csv(scan_path, index=False)
    logging.info(f"Scan complete: {len(missing_indices)} entries have missing files. Scan saved to {scan_path}")

    if args.scan_only:
        logging.info("--scan-only set; exiting after scan")
        return

    # ===== Phase 2: Check which indices have already been processed =====
    existing_chunk_files = sorted([f for f in os.listdir(chunks_dir) 
                                   if f.startswith("chunk_") and f.endswith(".csv")])
    processed_indices = set()
    
    for fname in existing_chunk_files:
        try:
            chunk_path = os.path.join(chunks_dir, fname)
            cdf = pd.read_csv(chunk_path)
            if "index" in cdf.columns:
                processed_indices.update(cdf["index"].tolist())
        except Exception as e:
            logging.exception(f"Failed to read existing chunk file {fname}: {e}")

    remaining = [idx for idx in missing_indices if idx not in processed_indices]
    logging.info(f"{len(processed_indices)} indices already processed. {len(remaining)} remaining to attempt.")

    if len(remaining) == 0:
        logging.info("All missing indices have been processed. Combining final report...")
    
    # ===== Phase 3: Process chunks of remaining indices =====
    chunk_size = args.chunk_size
    
    def partition(lst, n):
        """Partition list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    chunk_groups = list(partition(remaining, chunk_size))
    start_chunk_num = len(existing_chunk_files)
    
    for ci, group in enumerate(chunk_groups, start=start_chunk_num + 1):
        chunk_fname = os.path.join(chunks_dir, f"chunk_{ci:04d}.csv")
        
        if os.path.exists(chunk_fname):
            logging.info(f"Chunk {ci} already exists; skipping")
            continue

        logging.info(f"Processing chunk {ci} ({len(group)} indices: {group[0]} to {group[-1]})")
        chunk_rows = []
        
        for idx_in_chunk, index in enumerate(group):
            entry = df.loc[index]
            res = try_redownload(index, entry, out_dir, 
                                fov_deg=args.fov_deg, 
                                match_shape=True, 
                                max_attempts=args.max_attempts)
            
            chunk_rows.append({
                "index": index,
                "attempted": res["attempted"],
                "success": res["success"],
                "missing_LOFAR_before": res["missing_before"].get("LOFAR"),
                "missing_FIRST_before": res["missing_before"].get("FIRST"),
                "missing_NVSS_before": res["missing_before"].get("NVSS"),
                "missing_LOFAR_after": res["missing_after"].get("LOFAR"),
                "missing_FIRST_after": res["missing_after"].get("FIRST"),
                "missing_NVSS_after": res["missing_after"].get("NVSS"),
                "messages": " | ".join(res["messages"]) if res["messages"] else "",
            })
            
            if (idx_in_chunk + 1) % 10 == 0:
                logging.info(f"  Processed {idx_in_chunk + 1}/{len(group)} in chunk {ci}")

        # Write chunk results
        cdf = pd.DataFrame(chunk_rows)
        cdf.to_csv(chunk_fname, index=False)
        logging.info(f"Saved chunk {ci} results to {chunk_fname}")

    # ===== Phase 4: Combine all chunks into a final report =====
    logging.info("Combining all chunk results into final report...")
    all_chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) 
                              if f.startswith("chunk_") and f.endswith(".csv")])
    
    if all_chunk_files:
        combined_dfs = [pd.read_csv(f) for f in all_chunk_files]
        combined = pd.concat(combined_dfs, ignore_index=True)
        
        report_path = os.path.join(out_dir, "redownload_report.csv")
        combined.to_csv(report_path, index=False)
        logging.info(f"Final report saved to {report_path}")
        
        # Print summary stats
        n_total = len(combined)
        n_success = (combined["success"] == True).sum()
        n_still_missing = (combined["missing_LOFAR_after"] | combined["missing_FIRST_after"] | combined["missing_NVSS_after"]).sum()
        
        logging.info(f"\n=== SUMMARY ===")
        logging.info(f"Total indices checked: {n_total}")
        logging.info(f"Successfully downloaded: {n_success} ({100*n_success/n_total:.1f}%)")
        logging.info(f"Still missing files: {n_still_missing} ({100*n_still_missing/n_total:.1f}%)")
        logging.info(f"To retry failed downloads, simply rerun: python {__file__}")
    else:
        logging.info("No chunk files found; nothing to combine")

    logging.info("Done!")


if __name__ == "__main__":
    main()
