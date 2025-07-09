from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='kitchen_safety'
    )
    print("✅ التدريب انتهى! النموذج في مجلد runs/detect/kitchen_safety/weights")

if __name__ == '__main__':
    train_model()
