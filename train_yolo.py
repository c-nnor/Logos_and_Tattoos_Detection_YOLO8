from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="./dataset.yaml",  # Path to dataset.yaml
    epochs=50,                      # Number of training epochs
    imgsz=640,                      # Image size
    batch=16,                        # Batch size
    device="cuda"                    # Use GPU
)

# Save the trained model
model.export(format="onnx")


results = model.predict(source="test/images", save=True, conf=0.25)
