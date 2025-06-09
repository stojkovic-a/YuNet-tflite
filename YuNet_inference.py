import numpy as np
from collections import defaultdict
import cv2 as cv
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from yunet_tflite import YuNetTFLite


class YuNet(object):
    def __init__(self, model_path, **kwargs):
        self._model = YuNetTFLite(model_path, **kwargs)

    def inference(self, image):
        bboxes, landmarks, scores, (img_shape) = self._model.inference(image)
        bboxes = map(lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]], bboxes)
        image = cv.resize(image, (img_shape[2], img_shape[3]), cv.INTER_LINEAR)
        self._draw_boxes(image, bboxes, color=(0, 255, 0), label="face")
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow("inference", image)
        cv.waitKey(0)
        return (bboxes, landmarks, scores)

    def _calculate_padding(self, height, width, angle_rot):
        angle_rad = np.deg2rad(angle_rot)
        cos = abs(np.cos(angle_rad))
        sin = abs(np.sin(angle_rad))

        new_width = height * sin + width * cos
        new_height = height * cos + width * sin

        pad_x = int(np.ceil((new_width - width) / 2))
        pad_y = int(np.ceil((new_height - height) / 2))

        return pad_x, pad_y

    def _parse_annotations(self, annotations_file):
        labels = defaultdict(list)

        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                fname = parts[0]
                coords = list(map(int, parts[1:5]))  # x1 y1 x2 y2
                labels[fname].append(coords)

        return labels

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / float(area_box1 + area_box2 - intersection)

    def _draw_boxes(self, image, boxes, color, label):
        for box in boxes:
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv.putText(
                image,
                label,
                (box[0], box[1] - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def evaluate(
        self,
        images_dir,
        annotations_file,
        output_dir="output_results",
        iou_treshold=0.3,
        augment=True,
    ):
        os.makedirs(output_dir, exist_ok=True)

        total_true = 0
        total_pred = 0
        total_matched = 0
        ios_sum = 0
        iou_count = 0

        gt_labels = self._parse_annotations(annotations_file)

        for fname in os.listdir(images_dir):
            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping.")
                continue

            gt_boxes = gt_labels[fname]
            image = cv.imread(img_path)

            if augment:
                max_angle = 170

                pad_x, pad_y = self._calculate_padding(
                    image.shape[0], image.shape[1], max_angle
                )

                print(pad_x, pad_y)

                seq = iaa.Sequential(
                    [
                        iaa.Pad(px=((pad_y, pad_y, pad_x, pad_x)), keep_size=False),
                        iaa.Affine(rotate=(-max_angle, max_angle)),
                    ]
                )
                bbs = BoundingBoxesOnImage(
                    [
                        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        for x1, y1, x2, y2 in gt_boxes
                    ],
                    shape=image.shape,
                )

                image_aug, boxes_aug = seq(image=image, bounding_boxes=bbs)

                gt_boxes_aug = [
                    [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)]
                    for bb in boxes_aug.bounding_boxes
                ]

                image = image_aug
                gt_boxes = gt_boxes_aug

            bboxes, _, _ = self._model.inference(image)

            pred_boxes = []

            if bboxes is not None:
                for box in bboxes:
                    x1, y1, w, h = box.astype(int)
                    pred_boxes.append([x1, y1, x1 + w, y1 + h])

            matched_gt = set()
            matched_pred = set()

            for i, gt in enumerate(gt_boxes):
                best_iou = 0
                best_idx = -1

                for j, pred in enumerate(pred_boxes):
                    iou = self._compute_iou(gt, pred)
                    if iou >= iou_treshold and j not in matched_pred and iou > best_iou:
                        best_iou = iou
                        best_idx = j
                    if best_idx >= 0:
                        matched_gt.add(i)
                        matched_pred.add(best_idx)
                        ios_sum += best_iou
                        iou_count += 1

            total_true += len(gt_boxes)
            total_pred += len(pred_boxes)
            total_matched += len(matched_gt)

            image_out = image.copy()
            self._draw_boxes(image_out, gt_boxes, (0, 255, 0), "GT")
            self._draw_boxes(image_out, pred_boxes, (0, 0, 255), "pred")
            save_path = os.path.join(output_dir, fname)
            cv.imwrite(save_path, image_out)

        precision = total_matched / total_pred if total_pred > 0 else 0
        recall = total_matched / total_true if total_true > 0 else 0
        average_iou = ios_sum / iou_count if iou_count > 0 else 0

        metrics_file = os.path.join(
            output_dir, f"metrics_iouthd={str(iou_treshold)}.txt"
        )

        with open(metrics_file, "w") as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Average IoU: {average_iou:.4f}\n")


if __name__ == "__main__":
    net = YuNet("./YuNet-tflite/tflite_models/model.tflite")

    img = cv.imread("./YuNet-tflite/img.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    net.inference(img)
