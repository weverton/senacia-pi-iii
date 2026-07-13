from ultralytics import YOLO

# 1o estagio
model = YOLO("yolo26n.pt")
path = "./wider-dataset/data.yaml"
epochs = 5

model.train(data=path, epochs=10, freeze=10, name="wider_stage1_personal", exist_ok=True)

# 2o estagio
model = YOLO("/home/senacgoon.local/202473555/TecIA/runs/detect/wider_stage1_personal/weights/best.pt")

model.train(data=path, epochs=30, lr0=0.001, name="wider_stage2_personal", exist_ok=True)

# Validacao
model = YOLO("/home/senacgoon.local/202473555/TecIA/runs/detect/wider_stage2_personal/weights/best.pt")

results = model("./wider-dataset/WIDER Face Dataset For YOLOv12/WIDER Face Dataset For YOLOv12/test/images/*.jpg", stream=True, conf=0.74)

i = 0
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    print(f'info on {i}:\nQuantidade:{boxes.shape[0]}')
    result.save(filename=f"result{i}.jpg")  # save to disk
    i += 1    