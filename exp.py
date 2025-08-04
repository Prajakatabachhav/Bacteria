import os
import json
import torch
from yolox.exp import Exp as MyExp
from datetime import datetime

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.data_dir = r"D:\Internship Mareld\images"
        self.train_ann = r"D:\Internship Mareld\annotations\instances_train.json"
        self.val_ann = r"D:\Internship Mareld\annotations\instances_val.json"

        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = self.input_size
        self.max_epoch = 10
        self.batch_size = 1
        self.data_num_workers = 2
        self.seed = 42
        self.random_size = (14, 20)
        self.name = "custom_yolox_exp"
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.5
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.eval_interval = 5
        self.print_interval = 20
        self.enable_fast_coco_eval = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator
        from pycocotools.cocoeval import COCOeval
        import yolox.evaluators.coco_evaluator

        yolox.evaluators.coco_evaluator.COCOeval = COCOeval

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev=testdev),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

# =====================
# Main helpers for data conversion/visualization
# =====================
if __name__ == "__main__":
    import cv2
    from PIL import Image

    def yolo_to_coco(image_dir, label_dir, output_json_path, class_names):
        images = []
        annotations = []
        categories = []
        ann_id = 1

        for i, cls_name in enumerate(class_names, start=1):
            categories.append({"id": i, "name": cls_name, "supercategory": "none"})

        image_files = sorted([
            f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        for img_id, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

            if not os.path.exists(img_path):
                print(f"⚠️ Image not found: {img_path}")
                continue

            with Image.open(img_path) as img:
                w, h = img.size

            images.append({
                "id": img_id, "file_name": img_file, "width": w, "height": h
            })

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= len(class_names):
                        print(f"⚠️ Invalid class ID {cls_id} in {label_path}")
                        continue
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    bbox_width = float(parts[3]) * w
                    bbox_height = float(parts[4]) * h
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id + 1,  # <-- COCO expects 1-based category IDs
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    })
                    ann_id += 1

        coco_data = {
            "info": {
                "description": "Custom Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "",
                "date_created": datetime.today().strftime('%Y-%m-%d'),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"✅ COCO annotation saved at: {output_json_path}")

    def validate_coco_annotations(image_dir, annotation_path, num_classes):
        print(f"\n🔍 Validating {annotation_path}")
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        image_ids = {img['id'] for img in data['images']}
        category_ids = {cat['id'] for cat in data['categories']}

        for cat_id in category_ids:
            if cat_id < 0 or cat_id >= num_classes:
                print(f"⚠️ Category ID {cat_id} out of range")

        missing_img_ids = set()
        invalid_cat_ids = set()
        for ann in data['annotations']:
            if ann['image_id'] not in image_ids:
                missing_img_ids.add(ann['image_id'])
            if ann['category_id'] not in category_ids:
                invalid_cat_ids.add(ann['category_id'])

        if missing_img_ids:
            print(f"❌ Missing image IDs in 'images': {missing_img_ids}")
        if invalid_cat_ids:
            print(f"❌ Invalid category IDs in annotations: {invalid_cat_ids}")
        if not missing_img_ids and not invalid_cat_ids:
            print("✅ Annotations are internally consistent.")

        # Check image files exist
        missing_files = []
        for img in data['images']:
            img_path = os.path.join(image_dir, img['file_name'])
            if not os.path.exists(img_path):
                missing_files.append(img['file_name'])

        if missing_files:
            print(f"❌ Missing image files ({len(missing_files)}):")
            for mf in missing_files[:10]:
                print(f"   - {mf}")
        else:
            print("✅ All image files are present.")

    def visualize(image_path, annotations_json, save_dir=None, max_images=2,scale=0.5):
        with open(annotations_json, 'r') as f:
            data = json.load(f)

        count = 0
        for img_info in data['images']:
            if count >= max_images:
                break

            img_file = img_info['file_name']
            img_full_path = os.path.join(image_path, img_file)
            image = cv2.imread(img_full_path)
            if image is None:
                continue

            for ann in data['annotations']:
                if ann['image_id'] != img_info['id']:
                    continue
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, img_file), image)
            else:
                resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                cv2.imshow('bbox', resized)
                print(f"🖼️ Showing: {img_file}")
                cv2.waitKey(0)
                count += 1

        if not save_dir:
            cv2.destroyAllWindows()

    # === Example usage ===
    train_images_dir = r"D:\Internship Mareld\images\train2017"
    train_labels_dir = r"D:\Internship Mareld\label\sample-traintxt"
    val_images_dir = r"D:\Internship Mareld\images\val2017"
    val_labels_dir = r"D:\Internship Mareld\label\sample-testtxt"
    train_ann_path = r"D:\Internship Mareld\annotations\instances_train.json"
    val_ann_path = r"D:\Internship Mareld\annotations\instances_val.json"
    class_names = ["bacteria"]

    print("\n📁 Converting YOLO to COCO format...")
    yolo_to_coco(train_images_dir, train_labels_dir, train_ann_path, class_names)
    yolo_to_coco(val_images_dir, val_labels_dir, val_ann_path, class_names)

    print("\n🔍 Validating generated COCO JSONs...")
    validate_coco_annotations(train_images_dir, train_ann_path, len(class_names))
    validate_coco_annotations(val_images_dir, val_ann_path, len(class_names))

    print("\n🖼️ Visualizing annotations...")
    visualize(train_images_dir, train_ann_path, save_dir=None, max_images=2)

    print("\n✅ All preprocessing completed successfully!")
