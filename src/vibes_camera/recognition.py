"""Face recognition module for owner identification.

Uses face_recognition library to encode faces and compare against stored owner encoding.
Recognition is only run on cropped face regions from Haar cascade detections.
"""

import pickle
from pathlib import Path

import numpy as np

# Lazy import of face_recognition to avoid startup delay and warning message
# The module will be imported on first use
_face_recognition = None


def _get_face_recognition():
    """Lazily import face_recognition module."""
    global _face_recognition
    if _face_recognition is None:
        import face_recognition

        _face_recognition = face_recognition
    return _face_recognition


CONFIG_DIR = Path.home() / ".config" / "vibes_camera"
OWNER_FILE = CONFIG_DIR / "owner.pkl"


class FaceRecognizer:
    """Handles face encoding and owner recognition.

    Stores a single owner encoding and compares detected faces against it
    to determine if they are the owner or unknown.
    """

    def __init__(self, tolerance: float = 0.6):
        """Initialize the face recognizer.

        Args:
            tolerance: Distance threshold for face matching.
                      Lower values are stricter (fewer false positives).
                      Default 0.6 is recommended by face_recognition library.
        """
        self.tolerance = tolerance
        self.owner_encoding: np.ndarray | None = None
        self._load_owner()

    def _load_owner(self) -> bool:
        """Load owner encoding from disk.

        Returns:
            True if owner encoding was loaded, False if not found
        """
        if OWNER_FILE.exists():
            with open(OWNER_FILE, "rb") as f:
                self.owner_encoding = pickle.load(f)
            return True
        return False

    def save_owner(self, encodings: list[np.ndarray]) -> None:
        """Average encodings and save to disk.

        Args:
            encodings: List of 128-dim face encodings from enrollment
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.owner_encoding = np.mean(encodings, axis=0)
        with open(OWNER_FILE, "wb") as f:
            pickle.dump(self.owner_encoding, f)

    def encode_face(self, image: np.ndarray) -> np.ndarray | None:
        """Generate encoding from a cropped face image.

        Args:
            image: Cropped face region from Haar cascade (BGR format from OpenCV)

        Returns:
            128-dimensional face encoding, or None if no face could be encoded
        """
        # Convert BGR (OpenCV) to RGB (face_recognition)
        # The slice creates a non-contiguous array, but dlib (used by face_recognition)
        # requires contiguous arrays, especially with NumPy 2.x
        rgb_image = np.ascontiguousarray(image[:, :, ::-1])

        # Tell face_recognition where the face is located
        # This prevents encoding the WRONG face when the padded crop
        # includes a neighboring person's face
        # Format: (top, right, bottom, left)
        h, w = rgb_image.shape[:2]
        face_location = [(0, w, h, 0)]  # Face fills the whole crop

        fr = _get_face_recognition()
        encodings = fr.face_encodings(rgb_image, known_face_locations=face_location)

        if encodings:
            return encodings[0]
        return None

    def is_owner(self, encoding: np.ndarray) -> bool:
        """Check if encoding matches owner.

        Args:
            encoding: 128-dim face encoding to check

        Returns:
            True if encoding matches owner within tolerance
        """
        if self.owner_encoding is None:
            return False

        fr = _get_face_recognition()
        distance = fr.face_distance([self.owner_encoding], encoding)[0]
        return distance <= self.tolerance

    def has_owner(self) -> bool:
        """Check if owner is enrolled.

        Returns:
            True if owner encoding exists
        """
        return self.owner_encoding is not None

    @staticmethod
    def get_owner_file_path() -> Path:
        """Get the path to the owner encoding file.

        Returns:
            Path to owner.pkl file
        """
        return OWNER_FILE
