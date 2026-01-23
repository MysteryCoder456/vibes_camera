"""Basic camera capture application using OpenCV with face detection.

Uses multithreading to separate frame capture from face detection for better performance.
"""

import threading
import time

import cv2


# Configuration flags
USE_GPU = True  # Set to False to disable GPU acceleration (UMat)


def check_opencl_support():
    """Check and display OpenCL support status.

    Returns:
        bool: True if OpenCL is available and enabled, False otherwise.
    """
    print("\n=== OpenCL Status ===")

    have_opencl = cv2.ocl.haveOpenCL()
    print(f"OpenCL available: {have_opencl}")

    if not have_opencl:
        print("OpenCL is not available. GPU acceleration will not work.")
        print("This may be due to:")
        print("  - OpenCV was built without OpenCL support")
        print("  - No OpenCL-compatible device found")
        print("=====================\n")
        return False

    # Enable OpenCL if available
    cv2.ocl.setUseOpenCL(True)
    use_opencl = cv2.ocl.useOpenCL()
    print(f"OpenCL enabled: {use_opencl}")

    if use_opencl:
        try:
            device = cv2.ocl.Device.getDefault()
            print(f"Device name: {device.name()}")
            print(f"Device type: ", end="")
            dtype = device.type()
            if dtype == cv2.ocl.Device_TYPE_GPU:
                print("GPU")
            elif dtype == cv2.ocl.Device_TYPE_CPU:
                print("CPU")
            elif dtype == cv2.ocl.Device_TYPE_ACCELERATOR:
                print("Accelerator")
            else:
                print(f"Unknown ({dtype})")
            print(f"Device available: {device.available()}")
            print(f"OpenCL version: {device.OpenCLVersion()}")
        except Exception as e:
            print(f"Could not get device info: {e}")

    print("=====================\n")
    return use_opencl


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


def main():
    """Capture and display camera frames with face detection using Haar cascade."""
    # Check OpenCL support if GPU mode is enabled
    if USE_GPU:
        opencl_available = check_opencl_support()
        if not opencl_available:
            print("Warning: USE_GPU is True but OpenCL is not available.")
            print("Falling back to CPU mode.\n")

    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return 1

    # Load the Haar cascade classifiers for frontal and profile face detection
    frontal_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    if frontal_cascade.empty() or profile_cascade.empty():
        print("Error: Could not load Haar cascade classifiers")
        cap.release()
        return 1

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

    print("Camera opened successfully. Press 'q' to quit.")
    print("Detecting faces (frontal and profile) using Haar cascade classifiers...")
    print("Running with multithreading enabled.")
    print(f"GPU acceleration (UMat): {'enabled' if USE_GPU else 'disabled'}")

    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0

    # Main display loop
    while state.running:
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

        # Draw rectangles around detected faces (with mirrored coordinates)
        for x, y, w, h in detections:
            # Mirror the x coordinate
            mirrored_x = frame_width - x - w
            cv2.rectangle(
                display_frame, (mirrored_x, y), (mirrored_x + w, y + h), (0, 255, 0), 2
            )
            cv2.putText(
                display_frame,
                "Person",
                (mirrored_x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Calculate FPS
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        # Display the count of detected people and FPS
        cv2.putText(
            display_frame,
            f"People detected: {len(detections)}",
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

        # Display the frame
        cv2.imshow("Camera Feed - Face Detection", display_frame)

        # Wait for 1ms and check if 'q' was pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Signal threads to stop and wait for them
    state.stop()
    cap_thread.join(timeout=1.0)
    det_thread.join(timeout=1.0)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
