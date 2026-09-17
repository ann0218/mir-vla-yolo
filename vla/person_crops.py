#!/usr/bin/env python3
"""Turn one camera frame into the three streams the policy sees.

    camera1   the frame itself, with every tracked person boxed
    camera2   the LEFT person, cropped and blown up to full size
    camera3   the RIGHT person, same

Why this shape. The two-colour task stalled on resolution: a person occupied
1-7 of the 1024 patches SmolVLM2 splits a 512x512 input into, so the colour
naming them was a handful of tokens against a whole room. Measured on the
platform camera while the robot was actually following, a crop blown up to full
size carries 70,000-120,000 saturated pixels of one colour and none of the
other -- the two candidates become unmistakable, and the instruction only has to
choose between them.

Slots are ordered LEFT to RIGHT in the image, never by track id. Track ids are
handed out in order of first appearance, so the target -- acquired first,
because the expert starts by facing it -- would land in camera2 far more often
than chance, and the policy could score by always taking slot 2 without reading
anything. Left-to-right carries no such correlation, and it means the slot says
something true about where the person is, which the marked frame in camera1
agrees with.

The same code runs at conversion time and inside the policy server, so what the
model is trained on and what it is shown at inference cannot drift apart.
"""
import numpy as np

CROP_PAD = 8          # px of context around a box
BLANK = (32, 32, 32)  # what a missing person looks like


class PersonCrops:
    """Detector + tracker with per-episode state.

    One instance per episode or per closed-loop run: the tracker's ids only mean
    anything within a continuous sequence of frames, and reusing an instance
    across episodes carries stale tracks into the first frames of the next one.
    """

    def __init__(self, model_path="yolo11n.pt", conf=0.25, size=512,
                 max_people=2, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.size = size
        self.max_people = max_people
        self.device = device
        self.n = 0
        self.n_with = 0

    def _blank(self):
        img = np.empty((self.size, self.size, 3), dtype=np.uint8)
        img[:] = BLANK
        return img

    def __call__(self, bgr):
        """-> (marked, crops, boxes). crops is always max_people long."""
        import cv2
        h, w = bgr.shape[:2]
        r = self.model.track(bgr, persist=True, classes=[0], conf=self.conf,
                             tracker="bytetrack.yaml", verbose=False,
                             device=self.device)[0]

        boxes = []
        if r.boxes is not None and len(r.boxes):
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = [float(t) for t in r.boxes.xyxy[j]]
                tid = int(r.boxes.id[j]) if r.boxes.id is not None else -1
                boxes.append({"xyxy": (x1, y1, x2, y2), "id": tid,
                              "cx": (x1 + x2) / 2.0})
        # left to right, then keep the widest ones if a false positive appears
        boxes.sort(key=lambda b: b["cx"])
        if len(boxes) > self.max_people:
            keep = sorted(boxes, key=lambda b: -(b["xyxy"][2] - b["xyxy"][0])
                          )[:self.max_people]
            boxes = sorted(keep, key=lambda b: b["cx"])

        marked = bgr.copy()
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            xa, ya = max(0, int(x1) - CROP_PAD), max(0, int(y1) - CROP_PAD)
            xb, yb = min(w, int(x2) + CROP_PAD), min(h, int(y2) + CROP_PAD)
            cv2.rectangle(marked, (xa, ya), (xb, yb), (0, 255, 0), 2)
            sub = bgr[ya:yb, xa:xb]
            if sub.size == 0:
                crops.append(self._blank())
                continue
            crops.append(cv2.resize(sub, (self.size, self.size),
                                    interpolation=cv2.INTER_CUBIC))
        while len(crops) < self.max_people:
            crops.append(self._blank())

        self.n += 1
        if boxes:
            self.n_with += 1
        return marked, crops[:self.max_people], boxes

    def rate(self):
        return self.n_with / max(self.n, 1)
