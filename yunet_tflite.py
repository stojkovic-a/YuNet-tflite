#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from itertools import product

import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite


class YuNetTFLite(object):

    # Feature map
    MIN_SIZES = [[32], [64], [128]]
    STEPS = [8, 16, 32]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        model_path,
        input_shape=[640, 640],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
        num_threads=1,
    ):
        self.interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors
        self.priors = None
        self._generate_priors()

    def inference(self, image):
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            temp_image,
        )
        self.interpreter.invoke()

        bboxes = np.hstack(
            (
                self.interpreter.get_tensor(self.output_details[11]["index"]),
                self.interpreter.get_tensor(self.output_details[0]["index"]),
                self.interpreter.get_tensor(self.output_details[10]["index"]),
            )
        ).squeeze(axis=0)
        landmarks = np.hstack(
            (
                self.interpreter.get_tensor(self.output_details[9]["index"]),
                self.interpreter.get_tensor(self.output_details[6]["index"]),
                self.interpreter.get_tensor(self.output_details[3]["index"]),
            )
        ).squeeze(axis=0)
        obj = np.hstack(
            (
                self.interpreter.get_tensor(self.output_details[4]["index"]),
                self.interpreter.get_tensor(self.output_details[8]["index"]),
                self.interpreter.get_tensor(self.output_details[5]["index"]),
            )
        ).squeeze(axis=0)

        cls = np.hstack(
            (
                self.interpreter.get_tensor(self.output_details[2]["index"]),
                self.interpreter.get_tensor(self.output_details[1]["index"]),
                self.interpreter.get_tensor(self.output_details[7]["index"]),
            )
        ).squeeze(axis=0)

        bboxes, landmarks, scores = self._postprocess(bboxes, landmarks, obj, cls)

        return bboxes, landmarks, scores, (temp_image.shape)

    def _generate_priors(self):
        w, h = self.input_shape

        feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]

        feature_maps = [
            feature_map_3th,
            feature_map_4th,
            feature_map_5th,
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    def _preprocess(self, img):
        """image has to be RGB/BGR??????"""

        img_resized = cv.resize(
            img, (self.input_shape[0], self.input_shape[1]), cv.INTER_LINEAR
        )
        img_resized = img_resized.astype(np.float32)
        img_resized -= np.array([104, 117, 123])
        img_transposed = np.transpose(img_resized, (2, 0, 1))
        img_tensor = np.expand_dims(img_transposed, axis=0)
        return img_tensor

    def _postprocess(self, bboxes, landmarks, obj, cls):
        dets = self._decode(bboxes, landmarks, obj, cls)
        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # bboxes, landmarks, scores
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[: self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4].astype(np.int32))
                landmarks.append(det[4:14].astype(np.int32).reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, bbox, landmarks, obj, cls):
        _idx = np.where(cls < 0.0)
        cls[_idx] = 0.0

        _idx = np.where(obj < 0.0)
        obj[_idx] = 0.0

        _idx = np.where(cls > 1.0)
        cls[_idx] = 1.0

        _idx = np.where(obj > 1.0)
        obj[_idx] = 1.0

        scores = np.sqrt(cls * obj)

        scale = np.array(self.input_shape)

        # バウンディングボックス取得
        bboxes = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + bbox[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (self.priors[:, 2:4] * np.exp(bbox[:, 2:4] * self.VARIANCE)) * scale,
            )
        )
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # ランドマーク取得
        landmarks = np.hstack(
            (
                (
                    self.priors[:, 0:2]
                    + landmarks[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + landmarks[:, 2:4] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + landmarks[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + landmarks[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
                (
                    self.priors[:, 0:2]
                    + landmarks[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]
                )
                * scale,
            )
        )

        dets = np.hstack((bboxes, landmarks, scores))

        return dets


