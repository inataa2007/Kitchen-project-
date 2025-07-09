from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('runs/detect/kitchen_safety/weights/best.pt')

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(source=filename, save=False)
    objects = []
    for r in results:
        for box in r.boxes:
            obj = {
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "box": box.xywh[0].tolist()
            }
            objects.append(obj)
    return {"objects": objects}
