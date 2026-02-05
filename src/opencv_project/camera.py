"""Basic camera capture application using OpenCV with face detection.

Uses multithreading to separate frame capture from face detection for better performance.
"""

import argparse
import logging
import signal
import threading
import time

import cv2

from opencv_project.alerts import DiscordAlertManager, load_alert_config
from opencv_project.logging_config import setup_logging
from opencv_project.recognition import FaceRecognizer
from opencv_project.tracking import FaceTracker


# Configuration flags
USE_GPU = False  # Default to CPU mode; use --use-gpu flag to enable GPU acceleration

# Logger will be initialized in main()
logger: logging.Logger | None = None


def check_opencl_support():
    """Check and display OpenCL support status.

    Returns:
        bool: True if OpenCL is available and enabled, False otherwise.
    """
    logger.info("=== OpenCL Status ===")

    have_opencl = cv2.ocl.haveOpenCL()
    logger.info(f"OpenCL available: {have_opencl}")

    if not have_opencl:
        logger.warning("OpenCL is not available. GPU acceleration will not work.")
        logger.warning("This may be due to:")
        logger.warning("  - OpenCV was built without OpenCL support")
        logger.warning("  - No OpenCL-compatible device found")
        return False

    # Enable OpenCL if available
    cv2.ocl.setUseOpenCL(True)
    use_opencl = cv2.ocl.useOpenCL()
    logger.info(f"OpenCL enabled: {use_opencl}")

    if use_opencl:
        try:
            device = cv2.ocl.Device.getDefault()
            logger.info(f"Device name: {device.name()}")
            dtype = device.type()
            if dtype == cv2.ocl.Device_TYPE_GPU:
                device_type = "GPU"
            elif dtype == cv2.ocl.Device_TYPE_CPU:
                device_type = "CPU"
            elif dtype == cv2.ocl.Device_TYPE_ACCELERATOR:
                device_type = "Accelerator"
            else:
                device_type = f"Unknown ({dtype})"
            logger.info(f"Device type: {device_type}")
            logger.info(f"Device available: {device.available()}")
            logger.info(f"OpenCL version: {device.OpenCLVersion()}")
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")

    return use_opencl


def list_available_cameras(max_cameras=10):
    """Detect available cameras on the system.

    Args:
        max_cameras: Maximum number of camera indices to check

    Returns:
        List of (index, width, height, fps) tuples for available cameras
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            available.append((i, width, height, fps))
            cap.release()
    return available


def select_camera(cameras):
    """Display available cameras and let user select one.

    Args:
        cameras: List of (index, width, height, fps) tuples

    Returns:
        int: Selected camera index, or None if cancelled
    """
    logger.info(f"Found {len(cameras)} camera(s):")
    for idx, width, height, fps in cameras:
        logger.info(f"  [{idx}] Camera {idx}: {width}x{height} @ {fps:.0f} FPS")

    # Auto-select if only one camera
    if len(cameras) == 1:
        logger.info(f"Auto-selecting Camera {cameras[0][0]} (only one available)")
        return cameras[0][0]

    # Find default camera (prefer index 0 if available, else first in list)
    default_idx = cameras[0][0]
    for cam in cameras:
        if cam[0] == 0:
            default_idx = 0
            break

    # Prompt user
    valid_indices = [c[0] for c in cameras]
    while True:
        try:
            choice = input(f"Select camera [{default_idx}]: ").strip()

            # Default on empty input
            if choice == "":
                return default_idx

            choice_idx = int(choice)
            if choice_idx in valid_indices:
                return choice_idx
            logger.warning(f"Invalid selection. Choose from: {valid_indices}")
        except ValueError:
            logger.warning("Please enter a valid number.")
        except KeyboardInterrupt:
            logger.info("Cancelled.")
            return None


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Face detection application using Haar cascades"
    )
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=None,
        help="Camera index to use (skips interactive selection)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU acceleration (OpenCL)"
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Enable Discord DM alerts when people are detected",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Minimum seconds between alerts (default: 30)",
    )
    parser.add_argument(
        "--persistence",
        type=float,
        default=1.0,
        help="Seconds a face must be present before alerting (default: 1.0)",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Enroll owner face (captures 15 images over ~45 seconds)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Face recognition tolerance (lower=stricter, default: 0.6)",
    )
    parser.add_argument(
        "--owner-grace",
        type=int,
        default=300,
        help="Suppress alerts for N seconds after owner was last seen (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--display-feed",
        action="store_true",
        help="Display the camera feed window (default: headless mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


class SharedState:
    """Thread-safe container for sharing data between capture, detection, and display threads.

    Uses double buffering to eliminate frame copying overhead.
    """

    def __init__(self):
        # Double buffering for frames - swap references instead of copying
        self.buffers = [None, None]  # Two frame buffers
        self.write_idx = 0  # Index capture thread writes to
        self.buffer_lock = threading.Lock()

        self.detections = []  # Latest detection results
        self.detection_lock = threading.Lock()
        self.running = True  # Flag to stop threads

    def set_frame(self, frame):
        """Store frame in write buffer and swap buffers.

        The write buffer becomes the read buffer, and vice versa.
        This avoids copying the entire frame.
        """
        # Write to current write buffer
        self.buffers[self.write_idx] = frame
        # Swap indices atomically
        with self.buffer_lock:
            self.write_idx = 1 - self.write_idx

    def get_frame(self):
        """Get reference to the read buffer (no copy).

        Returns the buffer that is not currently being written to.
        """
        with self.buffer_lock:
            read_idx = 1 - self.write_idx
        return self.buffers[read_idx]

    def set_detections(self, detections):
        """Store the latest detection results."""
        with self.detection_lock:
            self.detections = detections

    def get_detections(self):
        """Get the latest detections (reference, not copy)."""
        with self.detection_lock:
            return self.detections

    def stop(self):
        """Signal all threads to stop."""
        self.running = False


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Tuple of (x, y, w, h) for first box
        box2: Tuple of (x, y, w, h) for second box

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def merge_detections(detections, iou_threshold=0.3):
    """Merge overlapping detections using Non-Maximum Suppression.

    Time complexity: O(n^2)

    Args:
        detections: List of (x, y, w, h) tuples
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        List of merged detections
    """
    if len(detections) == 0:
        return []

    # Convert to list of lists for easier manipulation
    boxes = [list(d) for d in detections]

    # Sort by area (larger boxes first)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue

        used[i] = True
        merged.append(tuple(box))

        # Mark overlapping boxes as used
        for j in range(i + 1, len(boxes)):
            if not used[j] and compute_iou(box, boxes[j]) > iou_threshold:
                used[j] = True

    return merged


def merge_detections_linear(detections, cell_size=50):
    """Merge overlapping detections using spatial hashing.

    Uses a grid-based approach where detections with centers in the same cell
    are merged together. This is faster but less accurate than IoU-based merging.

    Time complexity: O(n)

    Args:
        detections: List of (x, y, w, h) tuples
        cell_size: Size of grid cells for spatial hashing (in pixels).
                   Detections within the same cell are merged.

    Returns:
        List of merged detections
    """
    if len(detections) == 0:
        return []

    cells = {}  # key: (cell_x, cell_y) -> value: merged box

    for x, y, w, h in detections:
        # Calculate center of bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Get cell coordinates
        cell_key = (center_x // cell_size, center_y // cell_size)

        if cell_key in cells:
            # Merge with existing box (union of the two boxes)
            ex, ey, ew, eh = cells[cell_key]
            new_x = min(x, ex)
            new_y = min(y, ey)
            new_w = max(x + w, ex + ew) - new_x
            new_h = max(y + h, ey + eh) - new_y
            cells[cell_key] = (new_x, new_y, new_w, new_h)
        else:
            cells[cell_key] = (x, y, w, h)

    return list(cells.values())


def capture_thread(cap, state):
    """Continuously capture frames from the camera.

    Args:
        cap: OpenCV VideoCapture object
        state: SharedState object for storing frames
    """
    while state.running:
        ret, frame = cap.read()
        if ret:
            state.set_frame(frame)
        else:
            # Small sleep to avoid busy-waiting on error
            time.sleep(0.001)


def detection_thread(state, frontal_cascade, profile_cascade, detection_scale=0.5):
    """Continuously run face detection on the latest frame.

    Args:
        state: SharedState object for reading frames and storing detections
        frontal_cascade: Haar cascade for frontal face detection
        profile_cascade: Haar cascade for profile face detection
        detection_scale: Scale factor for resizing frame during detection
    """
    scale_inv = 1.0 / detection_scale
    scaled_min_size = (int(30 * detection_scale), int(30 * detection_scale))

    while state.running:
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.001)
            continue

        if USE_GPU:
            # GPU-accelerated path using UMat
            gpu_frame = cv2.UMat(frame)
            gpu_small = cv2.resize(
                gpu_frame, None, fx=detection_scale, fy=detection_scale
            )
            gpu_gray = cv2.cvtColor(gpu_small, cv2.COLOR_BGR2GRAY)
            small_width = int(frame.shape[1] * detection_scale)

            # Convert back to numpy for detectMultiScale (not GPU accelerated)
            gray = gpu_gray.get()

            # Flip on GPU for right-profile detection
            gpu_gray_flipped = cv2.flip(gpu_gray, 1)
            gray_flipped = gpu_gray_flipped.get()
        else:
            # CPU path
            small_frame = cv2.resize(
                frame, None, fx=detection_scale, fy=detection_scale
            )
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            small_width = small_frame.shape[1]
            gray_flipped = cv2.flip(gray, 1)

        # Detect frontal faces on smaller frame
        frontal_faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=scaled_min_size,
        )

        # Detect left-side profile faces on smaller frame
        left_profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=scaled_min_size,
        )

        # Detect right-side profile faces using flipped frame
        right_profiles_flipped = profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=scaled_min_size,
        )

        # Mirror the right profile coordinates back to original frame
        right_profiles = []
        for x, y, w, h in right_profiles_flipped:
            mirrored_x = small_width - x - w
            right_profiles.append((mirrored_x, y, w, h))

        # Combine all detections and scale coordinates back to original size
        all_detections = []
        for x, y, w, h in frontal_faces:
            all_detections.append(
                (
                    int(x * scale_inv),
                    int(y * scale_inv),
                    int(w * scale_inv),
                    int(h * scale_inv),
                )
            )
        for x, y, w, h in left_profiles:
            all_detections.append(
                (
                    int(x * scale_inv),
                    int(y * scale_inv),
                    int(w * scale_inv),
                    int(h * scale_inv),
                )
            )
        for x, y, w, h in right_profiles:
            all_detections.append(
                (
                    int(x * scale_inv),
                    int(y * scale_inv),
                    int(w * scale_inv),
                    int(h * scale_inv),
                )
            )

        # Merge overlapping detections to avoid double counting (linear time)
        merged_faces = merge_detections(all_detections)
        # merged_faces = merge_detections_linear(all_detections, cell_size=100)

        # Store detections
        state.set_detections(merged_faces)


def run_enrollment(cap, frontal_cascade, recognizer, tolerance):
    """Run the face enrollment process.

    Captures 15 images of the owner's face with 3-second intervals between each.
    Shows countdown timer and capture feedback on screen.

    Args:
        cap: OpenCV VideoCapture object
        frontal_cascade: Haar cascade for frontal face detection
        recognizer: FaceRecognizer instance
        tolerance: Recognition tolerance value

    Returns:
        0 on success, 1 on failure/cancellation
    """
    logger.info("=== Face Enrollment Mode ===")
    logger.info("Position your face in the camera frame.")
    logger.info("15 images will be captured with 3-second intervals.")
    logger.info("Press 'q' to cancel.")

    encodings = []
    target_captures = 15
    capture_interval = 3.0  # seconds between captures

    last_capture_time = time.time() - capture_interval  # Allow immediate first capture
    captures_done = 0

    while captures_done < target_captures:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Mirror frame for display
        display_frame = cv2.flip(frame, 1)
        frame_height, frame_width = display_frame.shape[:2]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(80, 80),
        )

        time_since_capture = time.time() - last_capture_time
        time_until_capture = max(0, capture_interval - time_since_capture)

        # Draw status text
        status_text = f"Enrollment: {captures_done}/{target_captures}"
        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        if len(faces) == 0:
            # No face detected
            cv2.putText(
                display_frame,
                "No face detected - position yourself in frame",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif len(faces) > 1:
            # Multiple faces detected
            cv2.putText(
                display_frame,
                "Multiple faces detected - only one person please",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            # Draw all face boxes in red
            for x, y, w, h in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            # Exactly one face - good!
            x, y, w, h = faces[0]

            # Draw face box
            if time_until_capture > 0:
                # Waiting - blue box with countdown
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 150, 0), 2)
                countdown_text = f"Capturing in: {time_until_capture:.1f}s"
                cv2.putText(
                    display_frame,
                    countdown_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 150, 0),
                    2,
                )
            else:
                # Ready to capture - crop BEFORE drawing with padding
                # Add 30% padding around face for better recognition
                pad_x = int(w * 0.3)
                pad_y = int(h * 0.3)
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(frame_width, x + w + pad_x)
                crop_y2 = min(frame_height, y + h + pad_y)
                face_crop = display_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                # Now draw green box (on original detection, not padded area)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Encode the clean crop
                encoding = recognizer.encode_face(face_crop)

                if encoding is not None:
                    encodings.append(encoding)
                    captures_done += 1
                    last_capture_time = time.time()
                    logger.info(f"Captured image {captures_done}/{target_captures}")

                    # Show capture feedback
                    cv2.putText(
                        display_frame,
                        f"Captured! ({captures_done}/{target_captures})",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Encoding failed - face_recognition couldn't find face in crop
                    cv2.putText(
                        display_frame,
                        "Encoding failed - hold still...",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        # Instructions at bottom
        cv2.putText(
            display_frame,
            "Press 'q' to cancel",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1,
        )

        cv2.imshow("Face Enrollment", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Enrollment cancelled.")
            cv2.destroyAllWindows()
            return 1

    # Save encodings
    logger.info(f"Captured {len(encodings)} images successfully!")
    logger.info("Saving owner encoding...")
    recognizer.save_owner(encodings)
    logger.info(f"Owner enrolled and saved to: {recognizer.get_owner_file_path()}")
    logger.info(f"Tolerance setting: {tolerance}")
    logger.info("You can now run with --recognize to enable face recognition.")

    cv2.destroyAllWindows()
    return 0


def main():
    """Capture and display camera frames with face detection using Haar cascade."""
    global USE_GPU, logger

    # Parse command-line arguments
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(level=log_level)

    # Signal handler for graceful shutdown (Ctrl+C)
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        logger.info("Shutdown requested (Ctrl+C)")
        shutdown_requested.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Enable GPU if --use-gpu flag is set
    if args.use_gpu:
        USE_GPU = True

    # Check OpenCL support if GPU mode is enabled
    if USE_GPU:
        opencl_available = check_opencl_support()
        if not opencl_available:
            logger.warning("GPU mode requested but OpenCL is not available.")
            logger.warning("Falling back to CPU mode.")
            USE_GPU = False

    # Camera selection
    logger.info("=== Camera Selection ===")

    if args.camera is not None:
        # Use camera specified via CLI argument
        camera_idx = args.camera
        logger.info(f"Using camera {camera_idx} (from --camera argument)")
    else:
        # Interactive selection
        logger.info("Scanning for available cameras...")
        cameras = list_available_cameras()

        if not cameras:
            logger.error("No cameras found!")
            return 1

        camera_idx = select_camera(cameras)
        if camera_idx is None:
            return 1

    logger.info(f"Opening Camera {camera_idx}...")

    # Open selected camera
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_idx}")
        return 1

    # Load the Haar cascade classifiers for frontal and profile face detection
    frontal_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    if frontal_cascade.empty() or profile_cascade.empty():
        logger.error("Could not load Haar cascade classifiers")
        cap.release()
        return 1

    # Handle enrollment mode separately
    if args.enroll:
        recognizer = FaceRecognizer(tolerance=args.tolerance)
        result = run_enrollment(cap, frontal_cascade, recognizer, args.tolerance)
        cap.release()
        return result

    # Initialize face recognizer - enabled automatically if owner is enrolled
    recognizer = FaceRecognizer(tolerance=args.tolerance)
    if recognizer.has_owner():
        logger.info("=== Face Recognition ===")
        logger.info(f"Owner loaded from: {recognizer.get_owner_file_path()}")
        logger.info(f"Recognition tolerance: {args.tolerance}")
        logger.info("Alerts will only trigger for unknown faces.")
    else:
        logger.info("=== No Owner Enrolled ===")
        logger.info("Face recognition disabled - no owner enrolled.")
        logger.info("Run with --enroll to set up face recognition.")
        logger.info("Alerts will trigger for all detected faces.")
        recognizer = None  # Disable recognition if no owner

    # Scale factor for resizing frame during detection (0.5 = half resolution)
    detection_scale = 0.5

    # Create shared state for communication between threads
    state = SharedState()

    # Start capture thread
    cap_thread = threading.Thread(
        target=capture_thread,
        args=(cap, state),
        daemon=True,
    )
    cap_thread.start()

    # Start detection thread
    det_thread = threading.Thread(
        target=detection_thread,
        args=(state, frontal_cascade, profile_cascade, detection_scale),
        daemon=True,
    )
    det_thread.start()

    # Initialize alert manager if alerts are enabled
    alert_manager = None
    if args.alerts:
        logger.info("=== Discord Alerts ===")
        config = load_alert_config()
        if config:
            try:
                logger.info("Connecting to Discord...")
                alert_manager = DiscordAlertManager(
                    config, cooldown_seconds=args.cooldown
                )
                logger.info(f"Discord DM alerts enabled (cooldown: {args.cooldown}s)")
            except Exception as e:
                logger.error(f"Failed to connect to Discord: {e}")
                logger.error("Exiting...")
                state.stop()
                cap.release()
                return 1
        else:
            logger.error("Alerts requested but credentials not configured.")
            logger.error("Exiting...")
            state.stop()
            cap.release()
            return 1

    # Initialize face tracker for persistence detection
    face_tracker = FaceTracker(persistence_threshold=args.persistence)

    logger.info(
        "Camera opened successfully. Press 'q' to quit (if display enabled) or Ctrl+C."
    )
    logger.info(
        "Detecting faces (frontal and profile) using Haar cascade classifiers..."
    )
    logger.info("Running with multithreading enabled.")
    logger.info(f"GPU acceleration (UMat): {'enabled' if USE_GPU else 'disabled'}")

    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0

    # Track when owner was last seen (for alert suppression)
    # Only triggers after owner is present for 5+ seconds consecutively
    last_owner_seen_time: float | None = None
    OWNER_PRESENCE_THRESHOLD = 5.0  # seconds owner must be present to trigger grace

    # Track which faces have been logged (to avoid duplicate logs per frame)
    # Maps face_id -> identity that was logged
    logged_faces: dict[int, str] = {}
    # Track previous frame's face IDs to detect when faces leave
    previous_face_ids: set[int] = set()
    # Track if we already logged alert suppression (to avoid spamming every frame)
    alert_suppression_logged = False
    # Track if grace period was logged for current owner presence (avoid spam)
    grace_period_logged_for_face: int | None = None

    # Main display loop
    while state.running and not shutdown_requested.is_set():
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.001)
            continue

        frame_width = frame.shape[1]

        # Mirror the frame for display
        if USE_GPU:
            # GPU-accelerated flip
            gpu_frame = cv2.UMat(frame)
            gpu_flipped = cv2.flip(gpu_frame, 1)
            display_frame = gpu_flipped.get()
        else:
            # CPU flip
            display_frame = cv2.flip(frame, 1)

        # Get latest detections and mirror their x coordinates
        detections = state.get_detections()

        # Mirror detections for display (since frame is mirrored)
        mirrored_detections = [
            (frame_width - x - w, y, w, h) for x, y, w, h in detections
        ]

        # Update face tracker with mirrored detections
        face_tracker.update(mirrored_detections)
        tracked_faces = face_tracker.get_all_faces()

        # Run face recognition on pending faces (only when recognizer is enabled)
        if recognizer:
            for face in face_tracker.get_pending_faces():
                x, y, w, h = face.bbox
                # Add 30% padding around face for better recognition
                # (face_recognition needs context beyond the tight Haar cascade crop)
                pad_x = int(w * 0.3)
                pad_y = int(h * 0.3)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(display_frame.shape[1], x + w + pad_x)
                y2 = min(display_frame.shape[0], y + h + pad_y)

                if x2 > x1 and y2 > y1:
                    face_crop = display_frame[y1:y2, x1:x2]
                    encoding = recognizer.encode_face(face_crop)

                    if encoding is not None:
                        if recognizer.is_owner(encoding):
                            face_tracker.set_identity(face.id, "owner")
                        else:
                            face_tracker.set_identity(face.id, "unknown")
                    # If encoding fails, keep as pending and retry next frame

        # Define colors for different states
        # BGR format: (Blue, Green, Red)
        COLOR_OWNER = (180, 200, 220)  # Beige
        COLOR_UNKNOWN = (0, 0, 255)  # Red
        COLOR_PENDING = (255, 150, 0)  # Blue-ish (recognition in progress)
        COLOR_PERSISTENT = (0, 255, 0)  # Green (no recognition mode)
        COLOR_NOT_PERSISTENT = (255, 150, 0)  # Blue (no recognition mode)

        # Track face states (needed for alerts even in headless mode)
        num_unknown = 0
        current_face_ids: set[int] = set()

        for face in tracked_faces:
            x, y, w, h = face.bbox
            current_face_ids.add(face.id)

            if recognizer:
                # Recognition mode: color based on identity
                if face.identity == "owner":
                    color = COLOR_OWNER
                    label = "Owner"

                    # Log when face is first identified as owner
                    if logged_faces.get(face.id) != "owner":
                        logger.info(
                            f"Face #{face.id} identified as OWNER "
                            f"(tracking for {face.duration:.1f}s)"
                        )
                        logged_faces[face.id] = "owner"

                    # Only update grace period if owner present for 5+ seconds
                    if face.duration >= OWNER_PRESENCE_THRESHOLD:
                        # Log grace period only once per owner face presence
                        if grace_period_logged_for_face != face.id:
                            logger.info(
                                f"Owner grace period activated - owner present for "
                                f"{face.duration:.1f}s, alerts paused for {args.owner_grace}s"
                            )
                            grace_period_logged_for_face = face.id
                        last_owner_seen_time = time.time()

                elif face.identity == "unknown":
                    color = COLOR_UNKNOWN
                    label = "Unknown"
                    num_unknown += 1

                    # Log when face is first identified as unknown
                    if logged_faces.get(face.id) != "unknown":
                        logger.info(
                            f"Face #{face.id} identified as UNKNOWN "
                            f"(tracking for {face.duration:.1f}s)"
                        )
                        logged_faces[face.id] = "unknown"

                else:  # pending
                    color = COLOR_PENDING
                    label = "..."
            else:
                # No recognition: color based on persistence
                is_persistent = face.is_persistent(args.persistence)
                color = COLOR_PERSISTENT if is_persistent else COLOR_NOT_PERSISTENT
                label = f"{face.duration:.1f}s"

            # Only draw if display is enabled OR alerts are enabled (for alert images)
            if args.display_feed or alert_manager:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Log faces that left the frame
        for face_id in previous_face_ids - current_face_ids:
            identity = logged_faces.pop(face_id, "unknown")
            logger.info(f"Face #{face_id} ({identity}) left frame")
            # Reset grace period log flag if this owner face left
            if face_id == grace_period_logged_for_face:
                grace_period_logged_for_face = None

        # Update previous face IDs for next iteration
        previous_face_ids = current_face_ids

        # Check and send alerts if enabled
        if alert_manager:
            # Check if owner was recently seen (suppress alerts during grace period)
            owner_recently_seen = (
                last_owner_seen_time is not None
                and (time.time() - last_owner_seen_time) < args.owner_grace
            )

            if recognizer:
                # Recognition mode: only alert for persistent unknown faces
                num_persistent_unknown = face_tracker.get_persistent_unknown_count()
                if alert_manager.should_alert(num_persistent_unknown):
                    if owner_recently_seen:
                        # Log suppressed alert (only once per suppression period)
                        if not alert_suppression_logged:
                            time_since = time.time() - last_owner_seen_time  # type: ignore
                            logger.info(
                                f"Alert suppressed - owner seen {time_since:.0f}s ago, "
                                f"{num_persistent_unknown} unknown face(s) detected"
                            )
                            alert_suppression_logged = True
                    else:
                        persistent_unknown = face_tracker.get_persistent_unknown_faces()
                        max_duration = (
                            max(f.duration for f in persistent_unknown)
                            if persistent_unknown
                            else 0
                        )
                        alert_manager.send_alert(
                            display_frame, num_persistent_unknown, max_duration
                        )
                        # Reset suppression flag when an alert is actually sent
                        alert_suppression_logged = False
                else:
                    # No alert needed, reset suppression flag for next time
                    alert_suppression_logged = False
                alert_manager.update_count(num_persistent_unknown)
            else:
                # No recognition: alert for all persistent faces
                num_persistent = face_tracker.get_persistent_count()
                if alert_manager.should_alert(num_persistent):
                    persistent_faces = face_tracker.get_persistent_faces()
                    max_duration = (
                        max(f.duration for f in persistent_faces)
                        if persistent_faces
                        else 0
                    )
                    alert_manager.send_alert(
                        display_frame, num_persistent, max_duration
                    )
                alert_manager.update_count(num_persistent)

        # Calculate FPS
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        # Draw status overlays (only if display is enabled)
        if args.display_feed:
            # Display status text
            if recognizer:
                num_persistent_unknown = face_tracker.get_persistent_unknown_count()
                status_text = (
                    f"Detected: {len(tracked_faces)} | "
                    f"Unknown: {num_unknown} | "
                    f"Persistent: {num_persistent_unknown}"
                )
            else:
                num_persistent = face_tracker.get_persistent_count()
                status_text = (
                    f"Detected: {len(tracked_faces)} | Persistent: {num_persistent}"
                )

            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Display owner grace period status (only in recognition mode with alerts)
            if recognizer and alert_manager:
                if last_owner_seen_time is not None:
                    time_since_owner = time.time() - last_owner_seen_time
                    if time_since_owner < args.owner_grace:
                        # Owner recently seen - alerts suppressed
                        remaining = int(args.owner_grace - time_since_owner)
                        mins, secs = divmod(remaining, 60)
                        grace_text = f"Owner seen {int(time_since_owner)}s ago - alerts paused ({mins}m {secs}s left)"
                        grace_color = (0, 180, 0)  # Green
                    else:
                        # Grace period expired - alerts active
                        mins_ago = int(time_since_owner // 60)
                        grace_text = f"Owner last seen {mins_ago}m ago - alerts active"
                        grace_color = (0, 165, 255)  # Orange
                else:
                    grace_text = "Owner not seen - alerts active"
                    grace_color = (0, 165, 255)  # Orange

                cv2.putText(
                    display_frame,
                    grace_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    grace_color,
                    1,
                )

            # Display the frame
            cv2.imshow("Camera Feed - Face Detection", display_frame)

            # Wait for 1ms and check if 'q' was pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Headless mode - small sleep to prevent CPU spinning
            time.sleep(0.001)

    # Signal threads to stop and wait for them
    state.stop()
    cap_thread.join(timeout=1.0)
    det_thread.join(timeout=1.0)

    # Shutdown alert manager if enabled
    if alert_manager:
        alert_manager.shutdown()

    # Release resources
    cap.release()
    if args.display_feed:
        cv2.destroyAllWindows()

    logger.info("Shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
