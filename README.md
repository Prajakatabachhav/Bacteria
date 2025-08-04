# Bacteria detection
This project sets up a for training YOLOX on a custom object detection dataset with YOLO-format annotations, converting them to COCO format, validating annotation integrity, and visualizing the results — all tailored for high-accuracy training on a single-class detection task (e.g., bacteria).

# Features
✅ YOLO to COCO format converter
Converts standard YOLO .txt annotations into COCO-style .json files for compatibility with YOLOX.

# Annotation validator
Ensures all image and annotation entries are consistent and complete (e.g., no missing files, invalid class IDs).

# Bounding box visualizer
Quickly view annotations on sample images using OpenCV to verify correctness before training.

# Custom YOLOX experiment configuration

Fully configurable Exp class

Single-class object detection

640×640 input size

Augmentations: mosaic, mixup, HSV, flip

Lightweight training setup for quick iteration (batch size 1, 10 epochs)
