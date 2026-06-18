#!/usr/bin/env python3
"""
GPMF telemetry extraction from a GoPro video.

IMPORTANT: the py_gpmf_parser library only works reliably on Linux. On Windows
the extraction fails, so this step must be run on Linux (or WSL, or a cluster).
The rest of the pipeline does work on any system.

It must be run on the ORIGINAL GoPro video (.MP4 without re-encoding): if the
video is re-exported or trimmed, the GPMF metadata is lost.

It generates a CSV with one row per frame that includes, at least, the columns
ACCL_x, ACCL_y, ACCL_z (accelerometer), which are the ones the coverage
calculation needs. Optionally it also extracts the embedded GPS (GPS5 stream).

Usage:
    python scripts/extract_telemetry.py --video GX010787.MP4 --out output/telemetry_frames.csv
    python scripts/extract_telemetry.py --video GX010787.MP4 --out output/telemetry_frames.csv --gps
"""
import argparse
import csv

import numpy as np

try:
    from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
except ImportError:
    raise SystemExit(
        "py_gpmf_parser was not found. Install it in a Linux environment:\n"
        "    pip install py-gpmf-parser\n"
        "This script does not work on Windows."
    )


def nearest_index(timestamps, t):
    """Index of the sample whose timestamp is closest to t."""
    return int(np.abs(timestamps - t).argmin())


def main():
    ap = argparse.ArgumentParser(description="Extracts GPMF telemetry from a GoPro video.")
    ap.add_argument("--video", required=True, help="Path to the ORIGINAL GoPro video (.MP4)")
    ap.add_argument("--out", required=True, help="Output CSV (telemetry_frames.csv)")
    ap.add_argument("--fps", type=float, default=30.0, help="Video FPS (default 30)")
    ap.add_argument("--gps", action="store_true",
                    help="Also extract the embedded GPS (GPS5 stream)")
    args = ap.parse_args()

    extractor = GoProTelemetryExtractor(args.video)
    extractor.open_source()

    # Accelerometer: essential to estimate the camera tilt
    accl, accl_ts = extractor.extract_data("ACCL")
    accl = np.asarray(accl)
    accl_ts = np.asarray(accl_ts)

    gps = gps_ts = None
    if args.gps:
        try:
            gps, gps_ts = extractor.extract_data("GPS5")
            gps = np.asarray(gps)
            gps_ts = np.asarray(gps_ts)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not extract GPS5 ({e}). Continuing without embedded GPS.")

    extractor.close_source()

    # Duration covered by the telemetry -> expected number of frames
    duration = float(accl_ts[-1])
    n_frames = int(duration * args.fps)

    fieldnames = ["frame_name", "timestamp_sec", "ACCL_x", "ACCL_y", "ACCL_z"]
    if gps is not None:
        fieldnames += ["GPMF_Latitude", "GPMF_Longitude"]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_frames):
            t = i / args.fps
            ai = nearest_index(accl_ts, t)
            row = {
                "frame_name": f"frame_{i:05d}.jpg",
                "timestamp_sec": round(t, 6),
                "ACCL_x": float(accl[ai][0]),
                "ACCL_y": float(accl[ai][1]),
                "ACCL_z": float(accl[ai][2]),
            }
            if gps is not None:
                gi = nearest_index(gps_ts, t)
                row["GPMF_Latitude"] = float(gps[gi][0])
                row["GPMF_Longitude"] = float(gps[gi][1])
            writer.writerow(row)

    print(f"Telemetry saved to {args.out} ({n_frames} frames)")
    if gps is not None:
        print("Includes embedded GPS (GPMF_Latitude, GPMF_Longitude).")


if __name__ == "__main__":
    main()
