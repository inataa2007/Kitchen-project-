# Kitchen Safety AI

🔍 مشروع ذكاء صناعي للكشف عن مخالفات HACCP و ISO 22000 باستخدام YOLOv8 + FastAPI.

## 🧪 التدريب
```bash
python train_yolov8.py
```

## 🚀 تشغيل الخادم
```bash
pip install -r requirements.txt
uvicorn ai_server:app --reload
```

ثم إرسال صورة عبر /analyze/ لتحليلها.
