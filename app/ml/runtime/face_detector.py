from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


class FaceDetectorUnavailable(RuntimeError):
    """Raised when the configured production face detector cannot run."""


class FaceDetectionError(ValueError):
    """Raised when detector output is not acceptable for runtime inference."""


@dataclass(frozen=True)
class FaceDetectionResult:
    status: str
    face_count: int
    bounding_boxes: list[dict[str, int]]
    detector: str = "opencv_haar_frontalface_default"

    @property
    def accepted(self) -> bool:
        return self.status == "one_face_detected"


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise FaceDetectorUnavailable("OpenCV face detector dependency is unavailable.") from exc
    return cv2


def detector_status() -> dict[str, object]:
    try:
        cv2 = _load_cv2()
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        available = cascade_path.exists()
        return {
            "available": available,
            "detector": "opencv_haar_frontalface_default",
            "failure_code": None if available else "CASCADE_NOT_FOUND",
            "version": getattr(cv2, "__version__", "unknown"),
        }
    except FaceDetectorUnavailable:
        return {
            "available": False,
            "detector": "opencv_haar_frontalface_default",
            "failure_code": "OPENCV_UNAVAILABLE",
            "version": None,
        }


def _detect_faces(path: Path) -> FaceDetectionResult:
    cv2 = _load_cv2()
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        raise FaceDetectorUnavailable("OpenCV Haar cascade file is unavailable.")
    image = cv2.imread(str(path))
    if image is None:
        raise FaceDetectionError("Face detector could not read the image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    boxes = [
        {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        for (x, y, w, h) in faces
    ]
    if len(boxes) == 0:
        return FaceDetectionResult(status="no_face_detected", face_count=0, bounding_boxes=[])
    if len(boxes) > 1:
        return FaceDetectionResult(status="multiple_faces_detected", face_count=len(boxes), bounding_boxes=boxes)
    return FaceDetectionResult(status="one_face_detected", face_count=1, bounding_boxes=boxes)


def detect_single_face(path: str | Path) -> FaceDetectionResult:
    source = Path(path)
    timeout = max(float(settings.FACE_DETECTOR_TIMEOUT_SECONDS), 0.1)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_detect_faces, source)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise FaceDetectionError("Face detector timed out.") from exc
    finally:
        if future.done():
            executor.shutdown(wait=True)
    if not result.accepted:
        raise FaceDetectionError(f"Face detector rejected image: {result.status}.")
    return result
