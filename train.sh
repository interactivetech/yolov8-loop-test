DATA_PATH=/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects-yolo
yolo task=detect \
  mode=train \
  model=yolov8s.pt \
  batch=32 \
  verbose=True \
  val=True \
  data=$DATA_PATH/data.yaml \
  epochs=2 \
  imgsz=224
