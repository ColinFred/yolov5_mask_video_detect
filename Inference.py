import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()
