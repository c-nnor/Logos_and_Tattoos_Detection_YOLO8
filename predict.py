from ultralytics import YOLO

#Route to best.pt required here
model = YOLO("runs/best.pt")

#This can point to a specific image or a directory
results = model.predict('example/dir/of/imgs', conf=0.25)

#run through the results of the model predictions
for result in results:
    result.show()