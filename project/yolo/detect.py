import cv2
import numpy as np
import time
from ultralytics import YOLO


class VisionDetector:
    """YOLOv8-based vision module with tracking, fire, and paper heuristics."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        fire_model_path: str | None = None,
        min_confidence: float = 0.25,
        detect_all_objects: bool = True,
        detection_profile: str = "balanced",
        person_only_mode: bool = False,
        person_threshold: float | None = None,
        object_threshold: float | None = None,
        paper_threshold: float | None = None,
        fire_threshold: float | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.fire_model = YOLO(fire_model_path) if fire_model_path else None
        self.min_confidence = max(0.05, min(0.95, float(min_confidence)))
        self.detect_all_objects = bool(detect_all_objects)
        self.detection_profile = str(detection_profile).strip().lower()
        self.person_only_mode = bool(person_only_mode)

        self.person_threshold = person_threshold
        self.object_threshold = object_threshold
        self.paper_threshold = paper_threshold
        self.fire_threshold = fire_threshold

        self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
        self.living_classes = {
            "person",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        }
        self.small_object_classes = {"cell phone", "remote", "book", "paper", "fire"}

        self.prev_gray = None
        self.fire_streak = 0
        self.fire_streak_required = 3

        self.track_state = {}
        self.next_track_id = 1
        self.track_ttl_s = 2.5
        self.track_max_distance = 140.0
        self.min_track_age_s = 0.5
        self.max_velocity_px_s = 1400.0
        self.track_lost_total = 0
        self.track_recovered_total = 0

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(1.0, area_a + area_b - inter)

    @staticmethod
    def _box_area(box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        return float(max(1, x2 - x1) * max(1, y2 - y1))

    def _nms_merge(self, detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        detections = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        kept = []
        for det in detections:
            suppress = False
            for prev in kept:
                if prev.get("class") != det.get("class"):
                    continue
                if self._iou(prev["bbox"], det["bbox"]) >= iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        return kept

    def _class_threshold(self, label: str) -> float:
        base = self.min_confidence
        if label == "person" and self.person_threshold is not None:
            return float(self.person_threshold)
        if label in {"paper", "book", "cell phone", "remote"} and self.paper_threshold is not None:
            return float(self.paper_threshold)
        if label == "fire" and self.fire_threshold is not None:
            return float(self.fire_threshold)
        if label != "person" and self.object_threshold is not None and label not in {"fire", "paper"}:
            return float(self.object_threshold)

        if self.detection_profile == "comprehensive":
            if label in {"cell phone", "remote", "book", "paper"}:
                return max(0.12, base - 0.08)
            if label in self.living_classes:
                return max(0.18, base)
            if label == "fire":
                return max(0.35, base + 0.05)
            if label in self.vehicle_classes or label == "vehicle":
                return max(0.22, base)
            return max(0.15, base - 0.05)

        if label == "person":
            return max(0.25, base)
        if label == "vehicle":
            return max(0.30, base + 0.05)
        if label == "fire":
            return max(0.45, base + 0.10)
        return max(0.45, base + 0.15)

    def _assign_tracks(self, detections: list[dict]) -> None:
        now = time.time()
        stale_ids = [
            tid
            for tid, data in self.track_state.items()
            if (now - float(data.get("last_seen", 0.0))) > self.track_ttl_s
        ]
        for tid in stale_ids:
            self.track_state.pop(tid, None)
            self.track_lost_total += 1

        used_track_ids = set()
        for det in detections:
            cls = str(det.get("class", ""))
            bbox = det["bbox"]
            cx, cy = self._centroid(bbox)

            best_id = None
            best_score = -1e9

            for tid, data in self.track_state.items():
                if tid in used_track_ids:
                    continue

                prev_bbox = data.get("bbox", bbox)
                px, py = data.get("centroid", (cx, cy))
                gap_s = now - float(data.get("last_seen", now))
                if gap_s > self.track_ttl_s:
                    continue

                dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
                dynamic_max_distance = self.track_max_distance * (1.0 + min(2.0, gap_s / max(0.2, self.track_ttl_s)))
                if dist > dynamic_max_distance:
                    continue

                dt = max(1e-3, gap_s)
                velocity = dist / dt
                if velocity > self.max_velocity_px_s:
                    continue

                iou = self._iou(prev_bbox, bbox)
                class_penalty = 0.0 if data.get("class") == cls else 0.35
                score = (1.0 - min(1.0, dist / max(1.0, dynamic_max_distance))) + 0.8 * iou - class_penalty
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.track_state[track_id] = {
                    "class": cls,
                    "bbox": bbox,
                    "centroid": (cx, cy),
                    "last_seen": now,
                    "first_seen": now,
                    "speed_px_s": 0.0,
                    "confidence_history": [float(det.get("confidence", 0.0))],
                    "motion_stability": 1.0,
                    "update_count": 1,
                    "lost_once": False,
                }
                det["track_id"] = track_id
                det["speed_px_s"] = 0.0
                det["tracked_seconds"] = 0.0
                det["track_confidence"] = 0.0
                used_track_ids.add(track_id)
                continue

            track = self.track_state[best_id]
            prev_cx, prev_cy = track.get("centroid", (cx, cy))
            prev_seen = float(track.get("last_seen", now))
            dt = max(1e-3, now - prev_seen)
            displacement = float((((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5))
            speed = displacement / dt
            continuity = max(0.0, 1.0 - min(1.0, displacement / max(1.0, self.track_max_distance)))

            prev_motion_stability = float(track.get("motion_stability", 1.0))
            motion_stability = 0.75 * prev_motion_stability + 0.25 * continuity

            confidence_history = list(track.get("confidence_history", []))
            confidence_history.append(float(det.get("confidence", 0.0)))
            confidence_history = confidence_history[-20:]
            conf_avg = float(sum(confidence_history) / max(1, len(confidence_history)))

            update_count = int(track.get("update_count", 1)) + 1

            if now - prev_seen > 0.6 and not track.get("lost_once", False):
                self.track_recovered_total += 1
                track["lost_once"] = True

            track["class"] = cls
            track["bbox"] = bbox
            track["centroid"] = (cx, cy)
            track["last_seen"] = now
            track["speed_px_s"] = speed
            track["motion_stability"] = motion_stability
            track["confidence_history"] = confidence_history
            track["update_count"] = update_count

            tracked_seconds = float(now - float(track.get("first_seen", now)))
            age_score = max(0.0, min(1.0, tracked_seconds / 3.0))
            track_confidence = 0.5 * conf_avg + 0.3 * motion_stability + 0.2 * age_score

            det["track_id"] = best_id
            det["speed_px_s"] = speed if tracked_seconds >= self.min_track_age_s else 0.0
            det["tracked_seconds"] = tracked_seconds
            det["track_confidence"] = float(max(0.0, min(1.0, track_confidence)))
            used_track_ids.add(best_id)

    def _detect_fire_with_model(self, frame: np.ndarray) -> list[dict]:
        if self.fire_model is None:
            return []

        detections = []
        result = self.fire_model(frame, verbose=False)[0]
        names = result.names
        for box in result.boxes:
            conf = float(box.conf.item())
            if conf < self._class_threshold("fire"):
                continue

            cls_idx = int(box.cls.item())
            label = str(names[cls_idx]).lower()
            if label not in {"fire", "flame", "smoke"}:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class": "fire",
                    "confidence": round(conf, 3),
                    "bbox": (x1, y1, x2, y2),
                    "source": f"fire_model:{label}",
                }
            )
        return detections

    def _motion_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, motion = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        motion = cv2.morphologyEx(motion, cv2.MORPH_DILATE, kernel)
        return motion

    def _detect_fire_regions(self, frame: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        motion = self._motion_mask(frame)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        lower_red1 = np.array([0, 120, 120], dtype=np.uint8)
        upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 120, 120], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
        lower_orange = np.array([12, 120, 120], dtype=np.uint8)
        upper_orange = np.array([35, 255, 255], dtype=np.uint8)

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        fire_mask = cv2.bitwise_or(mask_red, mask_orange)
        fire_mask = cv2.bitwise_and(fire_mask, cv2.bitwise_not(skin_mask))

        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)
        fire_motion_mask = cv2.bitwise_and(fire_mask, motion)

        contours, _ = cv2.findContours(fire_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(900, int(frame_area * 0.003))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            roi_fire = fire_mask[y : y + h, x : x + w]
            roi_motion = motion[y : y + h, x : x + w]
            roi_hsv = hsv[y : y + h, x : x + w]

            pix = float(w * h + 1e-6)
            fire_ratio = float(np.count_nonzero(roi_fire) / pix)
            motion_ratio = float(np.count_nonzero(roi_motion) / pix)
            sat_mean = float(np.mean(roi_hsv[:, :, 1]))
            val_mean = float(np.mean(roi_hsv[:, :, 2]))

            if fire_ratio < 0.18 or motion_ratio < 0.10 or sat_mean < 145 or val_mean < 135:
                continue

            confidence = 0.45 * fire_ratio + 0.40 * motion_ratio + 0.15 * min(1.0, sat_mean / 255.0)
            if confidence < self._class_threshold("fire"):
                continue

            detections.append(
                {
                    "class": "fire",
                    "confidence": round(min(0.99, confidence), 3),
                    "bbox": (x, y, x + w, y + h),
                    "source": "heuristic",
                }
            )

        strong_fire = any(d["confidence"] >= max(0.58, self._class_threshold("fire")) for d in detections)
        self.fire_streak = min(20, self.fire_streak + 1) if strong_fire else max(0, self.fire_streak - 1)
        if self.fire_streak < self.fire_streak_required:
            return []
        return detections

    def _detect_paper_regions(self, frame: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 55, 160)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.05:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) != 4:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw < 50 or bh < 50:
                continue

            rect_area = float(bw * bh)
            fill_ratio = area / max(1.0, rect_area)
            aspect_ratio = float(bw / max(1, bh))
            if fill_ratio < 0.55 or not (0.35 <= aspect_ratio <= 2.8):
                continue

            roi_hsv = hsv[y : y + bh, x : x + bw]
            sat_mean = float(np.mean(roi_hsv[:, :, 1]))
            val_mean = float(np.mean(roi_hsv[:, :, 2]))
            if sat_mean > 95 or val_mean < 85:
                continue

            area_ratio = area / frame_area
            confidence = 0.45 * min(1.0, area_ratio / 0.35) + 0.35 * fill_ratio + 0.20 * min(1.0, val_mean / 200.0)
            if confidence < self._class_threshold("paper"):
                continue

            candidates.append(
                {
                    "class": "paper",
                    "confidence": round(min(0.99, confidence), 3),
                    "bbox": (x, y, x + bw, y + bh),
                    "source": "heuristic",
                }
            )

        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        return candidates[:2]

    def _run_yolo(self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> list[dict]:
        output = []
        result = self.model(frame, verbose=False)[0]
        names = result.names
        frame_h, frame_w = frame.shape[:2]

        for box in result.boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = names[cls_idx]

            normalized_label = None
            if class_name == "person":
                normalized_label = "person"
            elif not self.person_only_mode and class_name in self.vehicle_classes and not self.detect_all_objects:
                normalized_label = "vehicle"
            elif not self.person_only_mode and self.detect_all_objects:
                normalized_label = class_name

            if normalized_label is None:
                continue
            if conf < self._class_threshold(normalized_label):
                continue

            if normalized_label == "person" and self.detection_profile != "comprehensive":
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area_ratio = float((bw * bh) / max(1, frame_w * frame_h))
                aspect = float(bw / bh)
                if area_ratio < 0.015 or aspect > 1.25:
                    continue

            output.append(
                {
                    "class": normalized_label,
                    "confidence": round(conf, 3),
                    "bbox": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    "source": "yolo",
                }
            )
        return output

    def _run_tiled_small_object_pass(self, frame: np.ndarray) -> list[dict]:
        if self.person_only_mode or not self.detect_all_objects:
            return []

        h, w = frame.shape[:2]
        if h < 240 or w < 320:
            return []

        tiles = [
            (0, 0, w // 2, h // 2),
            (w // 2, 0, w, h // 2),
            (0, h // 2, w // 2, h),
            (w // 2, h // 2, w, h),
            (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4),
        ]

        tile_detections = []
        for x1, y1, x2, y2 in tiles:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            for det in self._run_yolo(crop, offset_x=x1, offset_y=y1):
                label = str(det.get("class", ""))
                if label in self.small_object_classes:
                    tile_detections.append(det)
        return self._nms_merge(tile_detections, iou_threshold=0.35)

    def detect(self, frame: np.ndarray, meters_per_pixel: float | None = None) -> tuple[np.ndarray, list[dict]]:
        annotated = frame.copy()
        detections = self._run_yolo(frame)

        if self.detection_profile == "comprehensive":
            detections.extend(self._run_tiled_small_object_pass(frame))

        if not self.person_only_mode:
            fire_detections = self._detect_fire_with_model(frame)
            if not fire_detections:
                fire_detections = self._detect_fire_regions(frame)
            detections.extend(fire_detections)

            has_paper = any(d.get("class") == "paper" for d in detections)
            if not has_paper:
                detections.extend(self._detect_paper_regions(frame))

        detections = self._nms_merge(detections, iou_threshold=0.45)
        self._assign_tracks(detections)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            track_id = det.get("track_id")
            speed_px_s = float(det.get("speed_px_s", 0.0))
            speed_m_s = 0.0
            if meters_per_pixel is not None and meters_per_pixel > 0:
                speed_m_s = speed_px_s * float(meters_per_pixel)
            det["speed_m_s"] = speed_m_s

            tracked_s = float(det.get("tracked_seconds", 0.0))
            track_conf = float(det.get("track_confidence", 0.0))
            if speed_m_s > 0:
                label = (
                    f"{det['class'].upper()}#{track_id} {det['confidence']:.2f} "
                    f"{speed_px_s:.1f}px/s {speed_m_s:.2f}m/s {tracked_s:.1f}s tc={track_conf:.2f}"
                )
            else:
                label = (
                    f"{det['class'].upper()}#{track_id} {det['confidence']:.2f} "
                    f"{speed_px_s:.1f}px/s {tracked_s:.1f}s tc={track_conf:.2f}"
                )

            if det["class"] == "person":
                color = (80, 220, 80)
            elif det["class"] == "vehicle":
                color = (255, 200, 0)
            elif det["class"] == "paper":
                color = (255, 255, 0)
            else:
                color = (20, 60, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (x1 + 300, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated, detections