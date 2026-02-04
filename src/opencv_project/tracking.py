"""Face tracking for persistence detection.

Tracks faces across frames to determine how long each face has been present,
enabling alerts only for sustained presence rather than brief detections.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TrackedFace:
    """A face being tracked across frames."""

    id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        """How long this face has been tracked (seconds)."""
        return time.time() - self.first_seen

    @property
    def centroid(self) -> tuple[float, float]:
        """Center point of bounding box."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    def is_persistent(self, threshold: float) -> bool:
        """Check if face has been present longer than threshold.

        Args:
            threshold: Persistence threshold in seconds

        Returns:
            True if face duration exceeds threshold
        """
        return self.duration >= threshold


class FaceTracker:
    """Tracks faces across frames to determine persistence.

    Uses IoU (Intersection over Union) matching to associate detections
    across frames and track how long each face has been present.
    """

    def __init__(
        self,
        persistence_threshold: float = 1.0,
        max_missing_time: float = 0.3,
        iou_threshold: float = 0.3,
    ):
        """Initialize the face tracker.

        Args:
            persistence_threshold: Seconds a face must be present to be "persistent"
            max_missing_time: Seconds before removing a face that's no longer detected
            iou_threshold: Minimum IoU to consider two detections the same face
        """
        self.persistence_threshold = persistence_threshold
        self.max_missing_time = max_missing_time
        self.iou_threshold = iou_threshold
        self._tracked: dict[int, TrackedFace] = {}
        self._next_id: int = 0

    def update(self, detections: list[tuple[int, int, int, int]]) -> None:
        """Update tracker with new frame's detections.

        Matches new detections to existing tracked faces using IoU,
        creates new tracks for unmatched detections, and removes
        stale tracks that haven't been seen recently.

        Args:
            detections: List of (x, y, w, h) bounding boxes from current frame
        """
        now = time.time()

        # Match detections to existing tracked faces
        matched_track_ids: set[int] = set()
        unmatched_detections: list[tuple[int, int, int, int]] = []

        for det in detections:
            best_match_id = self._find_best_match(det, matched_track_ids)
            if best_match_id is not None:
                # Update existing track
                self._tracked[best_match_id].bbox = det
                self._tracked[best_match_id].last_seen = now
                matched_track_ids.add(best_match_id)
            else:
                unmatched_detections.append(det)

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            self._tracked[self._next_id] = TrackedFace(
                id=self._next_id,
                bbox=det,
                first_seen=now,
                last_seen=now,
            )
            self._next_id += 1

        # Remove stale tracks (faces not seen recently)
        stale_ids = [
            tid
            for tid, face in self._tracked.items()
            if now - face.last_seen > self.max_missing_time
        ]
        for tid in stale_ids:
            del self._tracked[tid]

    def _find_best_match(
        self,
        detection: tuple[int, int, int, int],
        already_matched: set[int],
    ) -> int | None:
        """Find best matching tracked face for a detection.

        Args:
            detection: (x, y, w, h) bounding box to match
            already_matched: Set of track IDs already matched this frame

        Returns:
            Track ID of best match, or None if no match found
        """
        best_iou = self.iou_threshold
        best_id = None

        for tid, face in self._tracked.items():
            if tid in already_matched:
                continue
            iou = self._calculate_iou(detection, face.bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = tid

        return best_id

    @staticmethod
    def _calculate_iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes.

        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)

        Returns:
            IoU value between 0.0 and 1.0
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection coordinates
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # No intersection
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_persistent_count(self) -> int:
        """Count faces present longer than persistence threshold.

        Returns:
            Number of persistent faces
        """
        return sum(
            1
            for face in self._tracked.values()
            if face.is_persistent(self.persistence_threshold)
        )

    def get_persistent_faces(self) -> list[TrackedFace]:
        """Get faces present longer than persistence threshold.

        Returns:
            List of TrackedFace objects that are persistent
        """
        return [
            face
            for face in self._tracked.values()
            if face.is_persistent(self.persistence_threshold)
        ]

    def get_all_faces(self) -> list[TrackedFace]:
        """Get all currently tracked faces.

        Returns:
            List of all TrackedFace objects
        """
        return list(self._tracked.values())
