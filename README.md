# Seagrass meadow coverage estimation in underwater video

A tool to **detect seagrass meadows and estimate their coverage** in underwater
video using a trained YOLO segmentation model, and to represent the result on a
**geolocated map**.

It works on **your own video**: it extracts the frames, runs the model, computes
the percentage of meadow coverage in each frame (correcting for camera tilt) and
generates an interactive map and a static image along the route.

This repository is the applied result of a Bachelor's Thesis (Data Science and
Engineering, EPI Gijón – University of Oviedo). The model was trained on meadows of
*Posidonia oceanica* and *Cymodocea nodosa* treated as a single functional category
of seagrass meadow.

---

## What do you need?

- An **underwater video** recorded with a camera that logs accelerometer telemetry
  (tested with a **GoPro Hero 9**, GPMF format).
- Optionally, **GPS** data to place the coverage on a map (see options).
- Python 3.10+.

---

## Installation

```bash
git clone https://github.com/<your_user>/posidonia-coverage.git
cd posidonia-coverage
pip install -r requirements.txt
```

Telemetry extraction uses `py-gpmf-parser`, which **only works on Linux**
(see below). Install it only in the Linux environment where you run that step:

```bash
pip install py-gpmf-parser
```

---

## Workflow

The process has **two phases**: one that requires Linux (telemetry) and another
that works on any system (the rest).

### Step 1 — Extract the telemetry (Linux ONLY)

The library that reads the GoPro's GPMF metadata does not work on Windows. This
step must be done on **Linux, WSL or a cluster**. It must be run on the
**original** camera video (without re-encoding or trimming; re-exporting loses the
telemetry).

```bash
python scripts/extract_telemetry.py \
    --video YOUR_VIDEO.MP4 \
    --out output/telemetry_frames.csv
```

Add `--gps` if you also want to extract the GPS that the GoPro itself records in
the video (see GPS options below):

```bash
python scripts/extract_telemetry.py --video YOUR_VIDEO.MP4 --out output/telemetry_frames.csv --gps
```

This generates `telemetry_frames.csv`, which the pipeline then uses. Once you have
that CSV, **the rest can be run on any operating system**.

> If your camera is not a GoPro or you do not want to use tilt correction, see the
> `tilt.enabled: false` option below.

### Step 2 — Run the pipeline

```bash
python scripts/run_pipeline.py \
    --config configs/config.yaml \
    --telemetry output/telemetry_frames.csv
```

It generates, in the output folder:

- `coverage_frames.csv` — coverage per frame.
- `coverage_map.html` — interactive map (if there is GPS).
- `coverage_map.png` — static map (if there is GPS).

---

## Configuration (`configs/config.yaml`)

All parameters are edited in that file. The most important ones:

### GPS source — `gps.source`

Choose according to the data you have:

| Value    | When to use it                                                            |
|----------|---------------------------------------------------------------------------|
| `gpx`    | You have an external `.gpx` file (e.g. from a tablet or phone).           |
| `gpmf`   | You want to use the GPS the GoPro records in the video (requires `--gps` in step 1). |
| `none`   | You have no GPS. Coverage is computed but **no** map is generated.         |

### GPS↔video time offset — `gps.offset_seconds`

Seconds of difference between the start of the GPS and the start of the video.

- **Leave it at `0`** if the video and the GPS start at the same time (normal case).
- Only change it if you **trimmed the video** relative to the original recording.
  For example, if you removed the first 95 s of the video but the GPX still starts
  at the original beginning, set `95`. (In the original TFG this value was 95.)

### Resolution — `frames.img_width` / `img_height`

The model was trained on **480×480** images. If you change the resolution, the
coverage estimation may degrade. Change it only if you know what you are doing.

### Tilt correction — `tilt`

The horizon line is estimated from the accelerometer in order to discard the water
column and compute coverage only over the visible seabed.

- `tilt.enabled: true` (recommended): corrects according to the camera tilt.
- `tilt.enabled: false`: uses the whole frame (use it if your camera always points
  at the seabed or if you do not have reliable accelerometer data).
- `angle_full_water` / `angle_no_water`: thresholds of the angle→horizon mapping.
  The default values (`-90` / `-60`) are calibrated for the TFG setup (camera on a
  sled, ~1 m above the seabed, pointing slightly forward). If your camera has a
  different orientation, adjust them.

---

## Limitations

- The coverage estimation is a **relative value**, not an absolute measure: the
  horizon line is a geometric approximation that assumes a flat seabed and a
  constant camera height, and it does not detect when tall vegetation blocks the
  horizon.
- The model was trained on a single route in the western Mediterranean; under very
  different conditions (other turbidity, depth, species) performance may vary.
- The model does not distinguish species: it treats all meadow as a single class.

---

## Repository structure

```
posidonia-coverage/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml              # all parameters
├── model/
│   └── best_2.pt                # trained model (YOLO segmentation)
├── scripts/
│   ├── extract_telemetry.py     # step 1 (Linux): GPMF telemetry
│   └── run_pipeline.py          # step 2: frames + GPS + inference + coverage + map

```

---

## Citation

If you use this tool, please cite the associated Bachelor's Thesis (Fernández
Vallés, C., University of Oviedo). See `LICENSE` for the terms of use.
