import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List
import logging
import warnings
from datetime import timedelta
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiVideoFaceRecognitionProcessor:
    def __init__(self, target_image_path: str, threshold: float = 0.6):
        self.threshold = threshold
        self.app = FaceAnalysis(det_name="retinaface_mnet025_v2", rec_name="arcface_r100_v1")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.target_embedding = self._get_target_embedding(target_image_path)
        self.camera_detected = [False] * 4
        self.trackers = [None] * 4
        self.tracker_frames = [0] * 4
        self.reinit_interval = 5
        self.detected_frames = [0] * 4  # Count frames where target is detected per camera
        self.time_spent = [timedelta(0) for _ in range(4)]  # Will be calculated at the end

    def _get_target_embedding(self, target_image_path: str) -> np.ndarray:
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            raise ValueError(f"Could not load target image: {target_image_path}")
        target_faces = self.app.get(target_image)
        if len(target_faces) == 0:
            raise ValueError("No face detected in the target image.")
        return target_faces[0].embedding

    def process_videos(self, video_paths: List[str], output_path: str = "outputs/combined_output.mp4"):
        caps = [cv2.VideoCapture(vp) for vp in video_paths]
        if not all(cap.isOpened() for cap in caps):
            raise ValueError("One or more video paths are invalid or could not be opened.")

        self.frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        scale = 0.4
        new_frame_width = int(self.frame_width * scale)
        new_frame_height = int(self.frame_height * scale)
        square_map_width = 800
        square_map_height = max(new_frame_height * 2, square_map_width)
        combined_width = new_frame_width * 2 + square_map_width
        combined_height = max(new_frame_height * 2, square_map_height)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*'mp4v'),
            fps,
            (combined_width, combined_height)
        )

        frame_count = 0
        window_name = "Multi-Camera Target Tracking with Enlarged Square Map"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width = min(combined_width // 2, 1200)
        display_height = int(display_width * combined_height / combined_width)
        cv2.resizeWindow(window_name, display_width, display_height)

        while True:
            frames = []
            ret_vals = []
            for cap in caps:
                ret, frame = cap.read()
                ret_vals.append(ret)
                frames.append(frame if ret else None)

            if not all(ret_vals):
                break
            frame_count += 1
            self.camera_detected = [False] * len(video_paths)

            processed_frames = []
            for idx, frame in enumerate(frames):
                processed_frame, match = self._process_frame(frame, idx)
                resized_frame = cv2.resize(processed_frame, (new_frame_width, new_frame_height))
                processed_frames.append(resized_frame)
                self.camera_detected[idx] = True if match else False
                if match:
                    self.detected_frames[idx] += 1  # Increment detected frame count

            highlighted_frames = []
            for idx, frame in enumerate(processed_frames):
                if self.camera_detected[idx]:
                    cv2.rectangle(frame, (0, 0), (new_frame_width, new_frame_height), (0, 255, 0), 4)
                highlighted_frames.append(frame)

            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_frame[0:new_frame_height, 0:new_frame_width] = highlighted_frames[0]
            combined_frame[0:new_frame_height, new_frame_width:new_frame_width*2] = highlighted_frames[1]
            combined_frame[new_frame_height:new_frame_height*2, 0:new_frame_width] = highlighted_frames[2]
            combined_frame[new_frame_height:new_frame_height*2, new_frame_width:new_frame_width*2] = highlighted_frames[3]

            square_visual = self._create_central_square(square_map_width, combined_height)
            combined_frame[0:combined_height, new_frame_width*2:combined_width] = square_visual

            status_text = f"Frames: {frame_count}"
            cv2.putText(combined_frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for idx, frames_detected in enumerate(self.detected_frames):
                time_seconds = frames_detected / fps
                time_text = f"Cam{idx+1}: {time_seconds:.1f}s"
                cv2.putText(combined_frame, time_text, (20, 80 + idx * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(combined_frame)
            cv2.imshow(window_name, combined_frame)

            delay = int(1000 / fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logger.info("Processing interrupted by user")
                break

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        # Calculate time spent per camera
        for idx in range(len(video_paths)):
            self.time_spent[idx] = timedelta(seconds=self.detected_frames[idx] / fps)

        for cap in caps:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

        logger.info("Video processing complete.")
        logger.info(f"Total iterations processed: {frame_count}")
        self._visualize_time_spent(fps)

    def _process_frame(self, frame: np.ndarray, cam_idx: int) -> tuple[np.ndarray, int]:
        match = 0
        tracker = self.trackers[cam_idx]

        if tracker is not None:
            try:
                if frame is None or not isinstance(frame, np.ndarray):
                    logger.error(f"Invalid frame for Camera {cam_idx + 1}")
                    self.trackers[cam_idx] = None
                else:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, "Tracking Target", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            match = 1
                        else:
                            logger.warning(f"Invalid bounding box for Camera {cam_idx + 1}: {bbox}")
                            self.trackers[cam_idx] = None
                    else:
                        logger.info(f"Tracker lost for Camera {cam_idx + 1}")
                        self.trackers[cam_idx] = None
            except Exception as e:
                logger.error(f"Tracker update failed for Camera {cam_idx + 1}: {e}")
                self.trackers[cam_idx] = None

        self.tracker_frames[cam_idx] += 1

        if self.trackers[cam_idx] is None or self.tracker_frames[cam_idx] >= self.reinit_interval:
            faces = self.app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                similarity = cosine_similarity([embedding], [self.target_embedding])[0][0]

                if similarity > self.threshold:
                    match = 1
                    self.trackers[cam_idx] = cv2.TrackerCSRT.create()
                    tracker_bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    try:
                        self.trackers[cam_idx].init(frame, tracker_bbox)
                        logger.info(f"Tracker initialized for Camera {cam_idx + 1} with similarity {similarity:.2f}")
                        self.tracker_frames[cam_idx] = 0
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"Target ({similarity:.2f})", (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
                    except Exception as e:
                        logger.error(f"Failed to initialize tracker for Camera {cam_idx + 1}: {e}")
                        self.trackers[cam_idx] = None
                        continue
                else:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"Unknown ({similarity:.2f})", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, match

    def _create_central_square(self, width: int, height: int) -> np.ndarray:
        square_visual = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = min(width, height) - 100
        center_x = width // 2
        center_y = height // 2
        half_size = square_size // 2

        top_left = (center_x - half_size, center_y - half_size)
        top_right = (center_x + half_size, center_y - half_size)
        bottom_left = (center_x - half_size, center_y + half_size)
        bottom_right = (center_x + half_size, center_y + half_size)

        cv2.rectangle(square_visual, top_left, bottom_right, (255, 255, 255), 4)
        camera_positions = [top_left, top_right, bottom_left, bottom_right]

        for idx, position in enumerate(camera_positions):
            color = (0, 255, 0) if self.camera_detected[idx] else (0, 0, 255)
            radius = 35 if self.camera_detected[idx] else 20
            cv2.circle(square_visual, position, radius, color, -1)
            cv2.putText(square_visual, f"Cam{idx+1}", (position[0] - 50, position[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return square_visual

    def _visualize_time_spent(self, fps: int):
        """Visualize time spent by the target in each camera as a bar graph."""
        time_in_seconds = [frames / fps for frames in self.detected_frames]
        cameras = [f"Camera {i+1}" for i in range(len(self.detected_frames))]

        plt.figure(figsize=(10, 6))
        plt.bar(cameras, time_in_seconds, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.xlabel('Cameras')
        plt.ylabel('Time Spent (seconds)')
        plt.title('Time Spent by Target in Each Camera')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(time_in_seconds):
            plt.text(i, v + 0.1, f"{v:.1f}s", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("time_spent_visualization.png")
        plt.show()
        logger.info("Time spent visualization saved as 'time_spent_visualization.png'")

def main():
    TARGET_IMAGE_PATH = "inputs/chandra_pic.jpg"
    VIDEO_PATHS = [
        "inputs/video_in_sandy.mp4",
        "inputs/video_in_uma.mp4",
        "inputs/video_in_rahul.mp4",
        "inputs/video_in_pixel.mp4"
    ]
    OUTPUT_PATH = "outputs/combined_output.mp4"
    THRESHOLD = 0.3

    processor = MultiVideoFaceRecognitionProcessor(TARGET_IMAGE_PATH, THRESHOLD)
    processor.process_videos(VIDEO_PATHS, OUTPUT_PATH)

if __name__ == "__main__":
    main()