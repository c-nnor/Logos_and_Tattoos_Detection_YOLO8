from ultralytics import YOLO

#Route to best.pt required here
# model = YOLO("runs/detect/train29/weights/best.pt")
model = YOLO("runs/detect/train10/weights/best.pt")

#This can point to a specific image or a directory
results = model.predict('test/images', conf=0.05)

# !main script for predicting on the test set!
#run through the results of the model predictions
# for result in results:
#     result.show()

# Example code for getting bounding box coordinates
for result in results:
    for box in result.boxes:
        # Get the class ID
        class_id = int(box.cls)
        # Convert the IDs to label names
        label = model.names[class_id]
        # Getting the confidence score
        confidence = float(box.conf)
        # Bounding box coords:D
        x_center, y_center, width, height = map(float, box.xywh[0])
        print(f"\nLabel:{label}\nConfidence: {confidence:.2f}\nCoordinates:  x_center: {x_center:.2f}"
              f" y_center: {y_center:.2f} width: {width:.2f} height:{height:.2f}")

