#!/usr/bin/env python3
"""
Seagrass meadow coverage estimation pipeline from underwater video.

Runs, in order:
  1. Frame extraction from the video
  2. GPS coordinate assignment (from an external GPX or from embedded GPMF)
  3. Segmentation inference with the trained model (best_2.pt)
  4. Per-frame coverage estimation (corrected for camera tilt)
  5. Geospatial visualization (interactive HTML map + static PNG)

Telemetry extraction (ACCL) is NOT done here: it must be generated beforehand with
scripts/extract_telemetry.py on Linux. See README.md.

Usage:
    python scripts/run_pipeline.py --config configs/config.yaml --telemetry output/telemetry_frames.csv
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
import gpxpy

from ultralytics import YOLO

import folium

import contextily as ctx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pyproj import Transformer


# =============================================================================
# 1. Frame extraction
# =============================================================================
def extract_frames(video_path, frames_dir, size, frame_skip):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {video_path}")

    frame_count = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, size)
            cv2.imwrite(str(frames_dir / f"frame_{saved:05d}.jpg"), frame)
            saved += 1
        frame_count += 1
    cap.release()
    print(f"[1/5] Frames extracted: {saved} -> {frames_dir}")
    return saved


# =============================================================================
# 2. Geolocation
# =============================================================================
def build_timestamps(frames_dir, fps):
    frames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
    return pd.DataFrame(
        {"frame_name": frames,
         "timestamp_sec": [round(i / fps, 6) for i in range(len(frames))]}
    )


def gps_from_gpx(ts_df, gpx_file, offset_seconds):
    """Assigns to each frame the nearest GPS point from an external GPX file."""

    with open(gpx_file, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    pts = [
        {"GPX_Time": p.time.replace(tzinfo=None),
         "Latitude": p.latitude, "Longitude": p.longitude}
        for tr in gpx.tracks for seg in tr.segments for p in seg.points
    ]
    gps_df = pd.DataFrame(pts)
    gps_df["GPX_Time"] = pd.to_datetime(gps_df["GPX_Time"], errors="coerce")

    # offset_seconds: offset if the video was trimmed relative to the GPS start
    video_start = gps_df["GPX_Time"].iloc[0] + pd.to_timedelta(offset_seconds, unit="s")
    gpx_times = gps_df["GPX_Time"].values.astype("int64")

    lat, lon = [], []
    for t in ts_df["timestamp_sec"]:
        frame_time = video_start + pd.to_timedelta(t, unit="s")
        j = int(np.abs(gpx_times - frame_time.value).argmin())
        lat.append(gps_df.iloc[j]["Latitude"])
        lon.append(gps_df.iloc[j]["Longitude"])

    out = ts_df.copy()
    out["GPS_Latitude"], out["GPS_Longitude"] = lat, lon
    return out


def gps_from_gpmf(ts_df, tel_df):
    """Uses the GPS embedded in the telemetry (GPMF_Latitude/Longitude columns)."""
    if not {"GPMF_Latitude", "GPMF_Longitude"}.issubset(tel_df.columns):
        raise ValueError(
            "The telemetry does not contain embedded GPS. Regenerate it with "
            "extract_telemetry.py --gps, or use gps.source: gpx / none."
        )
    out = ts_df.merge(
        tel_df[["frame_name", "GPMF_Latitude", "GPMF_Longitude"]],
        on="frame_name", how="left",
    ).rename(columns={"GPMF_Latitude": "GPS_Latitude",
                      "GPMF_Longitude": "GPS_Longitude"})
    return out


# =============================================================================
# 3. Inference
# =============================================================================
def run_inference(model_path, frames_dir, workdir, imgsz, conf, iou):

    model = YOLO(str(model_path))
    model.predict(
        source=str(frames_dir), imgsz=imgsz, conf=conf, iou=iou,
        save=False, save_txt=True, save_conf=True,
        project=str(workdir), name="pred", exist_ok=True,
    )
    labels_dir = workdir / "pred" / "labels"
    print(f"[3/5] Inference completed -> {labels_dir}")
    return labels_dir


# =============================================================================
# 4. Per-frame coverage
# =============================================================================
def tilt_to_pixel(tilt_deg, img_h, a_full, a_no):
    """Tilt angle -> vertical horizon coordinate (px)."""
    if tilt_deg < a_full:
        pct = 0.5
    elif tilt_deg > a_no:
        pct = 0.0
    else:
        pct = 0.5 - ((tilt_deg - a_full) / (a_no - a_full)) * 0.5
    return max(0, min(img_h - 1, int(pct * img_h)))


def estimate_coverage(geo_df, tel_df, labels_dir, cfg):
    img_w = cfg["frames"]["img_width"]
    img_h = cfg["frames"]["img_height"]
    tcfg = cfg["tilt"]

    df = geo_df.merge(
        tel_df[["frame_name", "ACCL_x", "ACCL_y", "ACCL_z"]],
        on="frame_name", how="left",
    )

    rows = []
    for _, row in df.iterrows():
        label_path = labels_dir / (os.path.splitext(row["frame_name"])[0] + ".txt")

        # Horizon line from the accelerometer (or disabled)
        if tcfg["enabled"] and not pd.isna(row.get("ACCL_x", np.nan)):
            tilt_rad = np.arctan2(-row["ACCL_x"],
                                  np.sqrt(row["ACCL_y"] ** 2 + row["ACCL_z"] ** 2))
            horizon_y = tilt_to_pixel(np.degrees(tilt_rad), img_h,
                                      tcfg["angle_full_water"], tcfg["angle_no_water"])
        else:
            horizon_y = 0  # no correction: the whole frame is used

        # Mask from normalized polygons
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    coords = list(map(float, line.split()[1:]))
                    if len(coords) % 2 != 0:        # save_conf adds an extra value
                        coords = coords[:-1]
                    pts = np.array(coords).reshape(-1, 2)
                    pts[:, 0] *= img_w
                    pts[:, 1] *= img_h
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

        below = mask[horizon_y:, :]
        total = int(below.size)
        cov = (int(below.sum()) / total * 100) if total else 0.0

        rec = {"frame_name": row["frame_name"],
               "timestamp_sec": row["timestamp_sec"],
               "horizon_y_px": horizon_y,
               "coverage_pct": round(cov, 4)}
        if "GPS_Latitude" in df.columns:
            rec["GPS_Latitude"] = row["GPS_Latitude"]
            rec["GPS_Longitude"] = row["GPS_Longitude"]
        rows.append(rec)

    print(f"[4/5] Coverage estimated for {len(rows)} frames")
    return pd.DataFrame(rows)


# =============================================================================
# 5. Map
# =============================================================================
def coverage_to_color_hex(pct):
    pct = max(0, min(100, pct))
    if pct <= 50:
        r, g, b = 255, int(255 * (pct / 50)), 0
    else:
        r, g, b = int(255 * (1 - (pct - 50) / 50)), 255, 0
    return f"#{r:02x}{g:02x}{b:02x}"


def make_maps(cov_df, workdir, window, zoom):
    if "GPS_Latitude" not in cov_df.columns or cov_df["GPS_Latitude"].isna().all():
        print("[5/5] No GPS: map generation skipped. "
              "Coverage CSV available anyway.")
        return


    df = cov_df.dropna(subset=["GPS_Latitude", "GPS_Longitude"]).reset_index(drop=True)
    df["group"] = df.index // window
    g = df.groupby("group").agg(
        GPS_Latitude=("GPS_Latitude", "mean"),
        GPS_Longitude=("GPS_Longitude", "mean"),
        coverage_pct=("coverage_pct", "mean"),
        frame_start=("frame_name", "first"),
        frame_end=("frame_name", "last"),
    ).reset_index()

    # --- Interactive HTML map ---
    mapa = folium.Map(location=[g["GPS_Latitude"].mean(), g["GPS_Longitude"].mean()],
                      zoom_start=zoom, max_zoom=20, tiles="Esri.WorldImagery")
    for i in range(len(g) - 1):
        a, b = g.iloc[i], g.iloc[i + 1]
        folium.PolyLine(
            [[a["GPS_Latitude"], a["GPS_Longitude"]],
             [b["GPS_Latitude"], b["GPS_Longitude"]]],
            color=coverage_to_color_hex((a["coverage_pct"] + b["coverage_pct"]) / 2),
            weight=5, opacity=0.9,
            tooltip=f"{a['frame_start']}→{a['frame_end']}: {a['coverage_pct']:.1f}%",
        ).add_to(mapa)
    legend = """
    <div style="position: fixed; bottom: 40px; right: 40px; z-index: 1000;
        background: white; padding: 15px 20px; border-radius: 8px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4); font-family: Arial; font-size: 13px;">
        <b>Meadow coverage</b><br><br>
        <div style="width:180px;height:18px;
            background:linear-gradient(to right,red,yellow,green);border-radius:4px;"></div>
        <div style="display:flex;justify-content:space-between;width:180px;">
            <span>0%</span><span>50%</span><span>100%</span></div>
    </div>"""
    mapa.get_root().html.add_child(folium.Element(legend))
    html_path = workdir / "coverage_map.html"
    mapa.save(str(html_path))

    # --- Static PNG ---
    try:

        cmap = mcolors.LinearSegmentedColormap.from_list("cov", ["red", "yellow", "green"])
        norm = mcolors.Normalize(vmin=0, vmax=100)
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        g["x"], g["y"] = tr.transform(g["GPS_Longitude"].values, g["GPS_Latitude"].values)

        fig, ax = plt.subplots(figsize=(12, 10))
        for i in range(len(g) - 1):
            x1, y1, c1 = g.iloc[i][["x", "y", "coverage_pct"]]
            x2, y2, c2 = g.iloc[i + 1][["x", "y", "coverage_pct"]]
            ax.plot([x1, x2], [y1, y2], color=cmap(norm((c1 + c2) / 2)),
                    linewidth=4, solid_capstyle="round")
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04).set_label("Coverage (%)")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(workdir / "coverage_map.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[5/5] Maps generated -> {html_path} and coverage_map.png")
    except Exception as e:  # noqa: BLE001
        print(f"[5/5] HTML map generated -> {html_path}. PNG skipped ({e}).")


# =============================================================================
# Orchestration
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Seagrass meadow coverage pipeline.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--telemetry", required=True,
                    help="Telemetry CSV generated with extract_telemetry.py")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    workdir = Path(cfg["paths"]["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "frames"
    size = (cfg["frames"]["img_width"], cfg["frames"]["img_height"])

    tel_df = pd.read_csv(args.telemetry)

    # 1. Frames
    extract_frames(Path(cfg["paths"]["video"]), frames_dir, size,
                   cfg["frames"]["frame_skip"])

    # 2. GPS
    ts_df = build_timestamps(frames_dir, cfg["frames"]["fps"])
    source = cfg["gps"]["source"]
    if source == "gpx":
        geo_df = gps_from_gpx(ts_df, cfg["gps"]["gpx_file"], cfg["gps"]["offset_seconds"])
        print(f"[2/5] GPS from GPX (offset {cfg['gps']['offset_seconds']} s)")
    elif source == "gpmf":
        geo_df = gps_from_gpmf(ts_df, tel_df)
        print("[2/5] GPS from embedded GPMF telemetry")
    elif source == "none":
        geo_df = ts_df
        print("[2/5] No GPS: coverage will be computed but no map")
    else:
        raise ValueError(f"Invalid gps.source: {source} (use gpx / gpmf / none)")

    # 3. Inference
    labels_dir = run_inference(Path(cfg["paths"]["model"]), frames_dir, workdir,
                               cfg["inference"]["imgsz"], cfg["inference"]["conf"],
                               cfg["inference"]["iou"])

    # 4. Coverage
    cov_df = estimate_coverage(geo_df, tel_df, labels_dir, cfg)
    cov_csv = workdir / "coverage_frames.csv"
    cov_df.to_csv(cov_csv, index=False)
    print(f"      Coverage CSV -> {cov_csv}")

    # 5. Map
    make_maps(cov_df, workdir, cfg["map"]["window"], cfg["map"]["zoom"])
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()