from ultralytics import YOLO

#Route to best.pt required here
# model = YOLO("runs/detect/train29/weights/best.pt")
model = YOLO("runs/detect/train36/weights/best.pt")

#This can point to a specific image or a directory
results = model.predict('test/images', conf=0.05)

#run through the results of the model predictions
for result in results:
    result.show()

