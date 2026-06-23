"""
CourtVision — Video Processing Backend
Run: py -m uvicorn main:app --reload --port 8000
"""

import cv2
import numpy as np
import tempfile
import os
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI(title="CourtVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8m.pt")

PALETTE = [
    (0, 229, 160),   (255, 107, 53),  (79, 195, 247),  (255, 213, 79),
    (206, 147, 216), (239, 154, 154), (128, 203, 196),  (255, 204, 2),
    (165, 214, 167), (255, 171, 64),  (100, 181, 246),  (240, 98, 146),
]

track_colors = {}
jobs = {}


def get_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = PALETTE[len(track_colors) % len(PALETTE)]
    return track_colors[track_id]


def is_on_pitch(x1, y1, x2, y2, frame_w, frame_h) -> bool:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return (
        cx > frame_w * 0.18 and
        cx < frame_w * 0.93 and
        cy > frame_h * 0.35 and
        cy < frame_h * 0.87 and
        h  > frame_h * 0.04 and
        h  < frame_h * 0.50 and   # ignore giant false positives
        w  < h * 1.2               # players are roughly taller than wide
    )


def draw_box(frame, x1, y1, x2, y2, track_id, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"#{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), font, scale, (0, 0, 0), thick)


@app.post("/process")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_input.write(await file.read())
    tmp_input.close()

    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output.close()

    tmp_data = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_data.close()

    job_id = os.path.basename(tmp_output.name).replace(".mp4", "")
    jobs[job_id] = {"status": "processing", "progress": 0}

    background_tasks.add_task(
        run_tracking, tmp_input.name, tmp_output.name, tmp_data.name, job_id
    )
    return {"job_id": job_id}


def run_tracking(input_path, output_path, data_path, job_id):
    try:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

        track_colors.clear()

        # track_data[track_id] = { color, frames: [{frame, bbox, cx, cy}], stats }
        track_data = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=0.45,
                iou=0.50,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if not is_on_pitch(x1, y1, x2, y2, frame_w, frame_h):
                        continue

                    track_id = int(box.id.item()) if box.id is not None else -1
                    if track_id == -1:
                        continue

                    color = get_color(track_id)
                    draw_box(frame, x1, y1, x2, y2, track_id, color)

                    # Save track data for the interactive player
                    if track_id not in track_data:
                        track_data[track_id] = {
                            "id": track_id,
                            "color": f"rgb{color}",
                            "frames": [],
                            "stats": {
                                "distance": 0.0,
                                "max_speed": 0.0,
                                "detections": 0,
                            }
                        }

                    td = track_data[track_id]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Calc speed from previous position
                    speed = 0.0
                    if td["frames"]:
                        prev = td["frames"][-1]
                        dx = cx - prev["cx"]
                        dy = cy - prev["cy"]
                        pixel_dist = np.sqrt(dx*dx + dy*dy)
                        # rough px→m factor — calibrate per video
                        metres = pixel_dist * (105 / frame_w) * 0.5
                        speed = metres / (1 / fps) * 3.6  # km/h
                        td["stats"]["distance"] += metres
                        td["stats"]["max_speed"] = max(td["stats"]["max_speed"], speed)

                    td["stats"]["detections"] += 1
                    td["frames"].append({
                        "frame": frame_idx,
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "cx": cx, "cy": cy,
                        "speed": round(speed, 1),
                    })

            out.write(frame)
            frame_idx += 1
            jobs[job_id]["progress"] = round(frame_idx / total_frames * 100)

        cap.release()
        out.release()

        # Round stats
        for td in track_data.values():
            td["stats"]["distance"] = round(td["stats"]["distance"], 1)
            td["stats"]["max_speed"] = round(td["stats"]["max_speed"], 1)
            td["stats"]["time_on_pitch"] = round(td["stats"]["detections"] / fps, 1)

        with open(data_path, "w") as f:
            json.dump({
                "fps": fps,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "total_frames": total_frames,
                "players": track_data,
            }, f)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["output"] = output_path
        jobs[job_id]["data"]   = data_path

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
    finally:
        os.unlink(input_path)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {"status": job["status"], "progress": job.get("progress", 0)}


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "Not ready"}
    return FileResponse(job["output"], media_type="video/mp4", filename="courtvision_tracked.mp4")


@app.get("/data/{job_id}")
async def get_data(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return {"error": "Not ready"}
    with open(job["data"]) as f:
        return JSONResponse(json.load(f))


@app.get("/health")
async def health():
    return {"status": "ok"}
