"""Evaluate pretrained COCO YOLO11n on the dataset test split with correct class mapping.

Dataset classes: 0 Ambulance, 1 Bus, 2 Car, 3 Motorcycle, 4 Truck
COCO ids used:   bus=5, car=2, motorcycle=3, truck=7
Ambulance has no COCO equivalent, so it is not evaluated.

Metric: precision and recall at IoU >= 0.5 and confidence >= 0.25,
with greedy matching per image. This is a simple custom metric, not the
Ultralytics mAP, so report it as such.
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "raw" / "cars-detection" / "test"
MODEL = ROOT / "models" / "yolo11n.pt"

DATASET_NAMES = {0: "Ambulance", 1: "Bus", 2: "Car", 3: "Motorcycle", 4: "Truck"}
COCO_TO_DATASET = {5: 1, 2: 2, 3: 3, 7: 4}
EVALUATED = [1, 2, 3, 4]
IOU_THRESH = 0.5
CONF = 0.25


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt(label_path, height, width):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height]
        boxes.append((cls, box))
    return boxes


def main():
    model = YOLO(str(MODEL))
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend((TEST_DIR / "images").glob(ext))
    images = sorted(images)
    print(f"Test images: {len(images)}")

    tp = {c: 0 for c in EVALUATED}
    fp = {c: 0 for c in EVALUATED}
    fn = {c: 0 for c in EVALUATED}

    for img in images:
        result = model.predict(str(img), conf=CONF,
                               classes=list(COCO_TO_DATASET), verbose=False)[0]
        height, width = result.orig_shape
        gt = load_gt(TEST_DIR / "labels" / f"{img.stem}.txt", height, width)

        preds = [(COCO_TO_DATASET[int(c)], float(s), b)
                 for b, c, s in zip(result.boxes.xyxy.tolist(),
                                    result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist())]

        for cls in EVALUATED:
            gts = [b for gc, b in gt if gc == cls]
            cls_preds = sorted([(s, b) for pc, s, b in preds if pc == cls],
                               key=lambda x: x[0], reverse=True)
            matched = [False] * len(gts)
            for _, box in cls_preds:
                best_j, best_iou = -1, IOU_THRESH
                for j, g in enumerate(gts):
                    if not matched[j]:
                        v = iou(box, g)
                        if v >= best_iou:
                            best_iou, best_j = v, j
                if best_j >= 0:
                    matched[best_j] = True
                    tp[cls] += 1
                else:
                    fp[cls] += 1
            fn[cls] += matched.count(False)

    print(f"\n{'Class':<12}{'GT':>6}{'Pred TP':>9}{'FP':>6}{'FN':>6}{'Precision':>11}{'Recall':>9}")
    p_list, r_list = [], []
    for cls in EVALUATED:
        gt_count = tp[cls] + fn[cls]
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / gt_count if gt_count else 0.0
        p_list.append(p)
        r_list.append(r)
        print(f"{DATASET_NAMES[cls]:<12}{gt_count:>6}{tp[cls]:>9}{fp[cls]:>6}{fn[cls]:>6}{p:>11.3f}{r:>9.3f}")

    print(f"{'Mean (4)':<12}{'':>6}{'':>9}{'':>6}{'':>6}{sum(p_list)/4:>11.3f}{sum(r_list)/4:>9.3f}")
    print("\nAmbulance: not evaluated (no COCO equivalent).")


if __name__ == "__main__":
    main()