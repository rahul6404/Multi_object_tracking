import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionProcessor:
    def __init__(self, target_image_path: str, threshold: float = 0.6):
        self.threshold = threshold
        self.tracker = None
        self.tracking = False

        # For handling occlusion gracefully:
        self.last_bbox = None
        self.lost_frames = 0
        self.lost_frames_threshold = 5  # Allow up to 5 consecutive lost frames

        # Initialize the InsightFace app
        self.app = FaceAnalysis(det_name="retinaface_mnet025_v2", rec_name="arcface_r100_v1")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load and process target image to obtain its embedding
        self.target_embedding = self._get_target_embedding(target_image_path)

    def _get_target_embedding(self, target_image_path: str) -> np.ndarray:
        """Extract embedding from target image."""
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            raise ValueError(f"Could not load target image: {target_image_path}")

        target_faces = self.app.get(target_image)
        if len(target_faces) == 0:
            raise ValueError("No face detected in the target image.")

        return target_faces[0].embedding

    def process_videos(self, video_paths: List[str], output_dir: str = "outputs"):
        """Process multiple videos and save results."""
        os.makedirs(output_dir, exist_ok=True)
        for video_path in video_paths:
            # Reset tracker for each new video.
            self.tracker = None
            self.tracking = False
            self.last_bbox = None
            self.lost_frames = 0

            logger.info(f"Processing video: {video_path}")
            self.process_single_video(video_path, output_dir)
            logger.info(f"Completed processing video: {video_path}")
            cv2.destroyAllWindows()

    def process_single_video(self, video_path: str, output_dir: str):
        """Process a single video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_filename = os.path.join(
            output_dir,
            f"processed_{os.path.basename(video_path)}"
        )
        out = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter.fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

        frame_count = 0
        matches_found = 0

        window_name = f"Processing: {os.path.basename(video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width = min(frame_width // 2, 800)
        display_height = int(display_width * frame_height / frame_width)
        cv2.resizeWindow(window_name, display_width, display_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # If tracking is active, update the tracker.
            if self.tracking:
                ok, bbox = self.tracker.update(frame)
                if ok:
                    # Reset lost_frames count if tracking is successful
                    self.lost_frames = 0
                    bbox = tuple(map(int, bbox))
                    self.last_bbox = bbox  # Store last known bbox
                    cv2.rectangle(frame,
                                  (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Tracker update failed, increase lost_frames counter.
                    self.lost_frames += 1
                    logger.info(f"Tracker update failed for {self.lost_frames} frame(s).")
                    # If we have a stored last bbox, draw it with a 'lost' message.
                    if self.last_bbox is not None:
                        cv2.rectangle(frame,
                                      (self.last_bbox[0], self.last_bbox[1]),
                                      (self.last_bbox[0] + self.last_bbox[2],
                                       self.last_bbox[1] + self.last_bbox[3]),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "Lost (waiting)", (self.last_bbox[0], self.last_bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # If lost for too many frames, give up tracking and revert to detection.
                    if self.lost_frames >= self.lost_frames_threshold:
                        logger.info("Tracker lost target for too many frames; reverting to detection.")
                        self.tracking = False
                        self.last_bbox = None
                        self.lost_frames = 0
            else:
                # Detection mode: run face detection on this frame.
                processed_frame, matches = self._process_frame(frame)
                matches_found += matches
                frame = processed_frame

            # Overlay status text.
            status_text = f"Frames: {frame_count} | Matches: {matches_found}"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            cv2.imshow(window_name, frame)

            delay = int(1000 / fps)  # delay in milliseconds according to the video fps
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logger.info("Processing interrupted by user")
                break

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames for {video_path}")

        cap.release()
        out.release()
        cv2.destroyWindow(window_name)

        logger.info(f"Video processing complete: {video_path}")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total matches found: {matches_found}")

        return output_filename

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Run face detection in the frame.
        If the target is found (based on cosine similarity), initialize (or reinitialize)
        the MOSSE tracker using a properly formatted ROI.
        """
        faces = self.app.get(frame)
        matches = 0

        for face in faces:
            bbox = face.bbox.astype(int)  # [x_min, y_min, x_max, y_max]
            embedding = face.embedding

            similarity = cosine_similarity([embedding], [self.target_embedding])[0][0]

            if similarity > self.threshold:
                matches += 1
                color = (0, 255, 0)
                label = f"Target (Sim: {similarity:.2f})"

                # If not already tracking, initialize (or reinitialize) the tracker.
                if not self.tracking:
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    if width > 0 and height > 0:
                        tracker_roi = (x_min, y_min, width, height)
                        # Use the legacy API for MOSSE tracker (ensure you have opencv-contrib installed)
                        self.tracker = cv2.legacy.TrackerMOSSE.create()
                        initialized = self.tracker.init(frame, tracker_roi)
                        if initialized:
                            self.tracking = True
                            self.last_bbox = tracker_roi
                            self.lost_frames = 0
                            logger.info(f"Tracker initialized with ROI: {tracker_roi}")
                        else:
                            logger.warning("Tracker failed to initialize.")
                    else:
                        logger.warning("Invalid bounding box dimensions, skipping tracker initialization.")
            else:
                color = (0, 0, 255)
                label = f"Unknown (Sim: {similarity:.2f})"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, matches


def main():
    TARGET_IMAGE_PATH = "inputs/chandra_pic.jpg"
    VIDEO_PATHS = [
        "inputs/video_in_pixel.mp4",
        "inputs/video_in_rahul.mp4",
        "inputs/video_in_sandy.mp4",
        "inputs/video_in_uma.mp4"
    ]
    OUTPUT_DIR = "outputs"
    THRESHOLD = 0.3

    processor = FaceRecognitionProcessor(TARGET_IMAGE_PATH, THRESHOLD)
    processor.process_videos(VIDEO_PATHS, OUTPUT_DIR)


if __name__ == "__main__":
    main()
