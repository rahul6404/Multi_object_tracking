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


class MultiVideoFaceRecognitionProcessor:
    def __init__(self, target_image_path: str, threshold: float = 0.6):
        self.threshold = threshold

        # Initialize the InsightFace app
        self.app = FaceAnalysis(det_name="retinaface_mnet025_v2", rec_name="arcface_r100_v1")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load and process target image to obtain its embedding
        self.target_embedding = self._get_target_embedding(target_image_path)

        # Initialize camera detection status
        self.camera_detected = [False] * 4  # One for each camera

    def _get_target_embedding(self, target_image_path: str) -> np.ndarray:
        """Extract embedding from target image."""
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            raise ValueError(f"Could not load target image: {target_image_path}")

        target_faces = self.app.get(target_image)
        if len(target_faces) == 0:
            raise ValueError("No face detected in the target image.")

        return target_faces[0].embedding

    def process_videos(self, video_paths: List[str], output_path: str = "outputs/combined_output.mp4"):
        """Process multiple videos and display them in an L-shape with an enlarged square map."""
        # Open all video captures
        caps = [cv2.VideoCapture(vp) for vp in video_paths]
        if not all(cap.isOpened() for cap in caps):
            raise ValueError("One or more video paths are invalid or could not be opened.")

        # Get properties (assuming all videos have the same properties)
        self.frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        # Resize factor for video frames to make them smaller
        scale = 0.4  # Adjust scaling factor as needed
        new_frame_width = int(self.frame_width * scale)
        new_frame_height = int(self.frame_height * scale)

        # Define the size of the square map visualization
        square_map_width = 800  # Adjust as needed
        square_map_height = max(new_frame_height * 2, square_map_width)

        # Combined frame dimensions
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

            # Reset camera detection status for this frame
            self.camera_detected = [False] * len(video_paths)

            # Process each frame individually
            processed_frames = []
            for idx, frame in enumerate(frames):
                processed_frame, match = self._process_frame(frame)
                # Resize the processed frame
                resized_frame = cv2.resize(processed_frame, (new_frame_width, new_frame_height))
                processed_frames.append(resized_frame)
                self.camera_detected[idx] = True if match else False

            # Highlight frames where the target is detected
            highlighted_frames = []
            for idx, frame in enumerate(processed_frames):
                if self.camera_detected[idx]:
                    # Add a green border to highlight
                    cv2.rectangle(frame, (0, 0), (new_frame_width, new_frame_height), (0, 255, 0), 4)
                highlighted_frames.append(frame)

            # Create a blank canvas for the combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # Arrange frames in L-shape

            # Position of Video1 (top-left)
            combined_frame[0:new_frame_height, 0:new_frame_width] = highlighted_frames[0]

            # Position of Video2 (to the right of Video1)
            combined_frame[0:new_frame_height, new_frame_width:new_frame_width*2] = highlighted_frames[1]

            # Position of Video3 (below Video1)
            combined_frame[new_frame_height:new_frame_height*2, 0:new_frame_width] = highlighted_frames[2]

            # Position of Video4 (optional, below Video2)
            combined_frame[new_frame_height:new_frame_height*2, new_frame_width:new_frame_width*2] = highlighted_frames[3]

            # Create the enlarged square visualization
            square_visual = self._create_central_square(square_map_width, combined_height)

            # Place the square visualization into combined frame
            combined_frame[0:combined_height, new_frame_width*2:combined_width] = square_visual

            # Overlay status text
            status_text = f"Frames Processed: {frame_count}"
            cv2.putText(combined_frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            out.write(combined_frame)
            cv2.imshow(window_name, combined_frame)

            delay = int(1000 / fps)  # Delay in milliseconds according to the video fps
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logger.info("Processing interrupted by user")
                break

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        for cap in caps:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

        logger.info("Video processing complete.")
        logger.info(f"Total frames processed: {frame_count}")

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Run face detection on the frame.
        Return the processed frame and a flag indicating if the target was found.
        """
        faces = self.app.get(frame)
        match = 0  # 0 if no match, 1 if match found

        for face in faces:
            bbox = face.bbox.astype(int)  # [x_min, y_min, x_max, y_max]
            embedding = face.embedding

            similarity = cosine_similarity([embedding], [self.target_embedding])[0][0]

            if similarity > self.threshold:
                match = 1
                color = (0, 255, 0)
                label = f"Target ({similarity:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({similarity:.2f})"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, match

    def _create_central_square(self, width: int, height: int) -> np.ndarray:
        """
        Create the enlarged square map visualization with camera indicators.
        Highlight cameras where the target is detected.
        """
        # Create a blank image for the square visualization
        square_visual = np.zeros((height, width, 3), dtype=np.uint8)

        # Square coordinates
        square_size = min(width, height) - 100  # Leave some padding
        center_x = width // 2
        center_y = height // 2

        half_size = square_size // 2

        # Square corners
        top_left = (center_x - half_size, center_y - half_size)
        top_right = (center_x + half_size, center_y - half_size)
        bottom_left = (center_x - half_size, center_y + half_size)
        bottom_right = (center_x + half_size, center_y + half_size)

        # Draw the square
        cv2.rectangle(square_visual, top_left, bottom_right, (255, 255, 255), 4)

        # Camera positions
        camera_positions = [top_left, top_right, bottom_left, bottom_right]

        # Draw camera indicators
        for idx, position in enumerate(camera_positions):
            color = (0, 255, 0) if self.camera_detected[idx] else (0, 0, 255)
            radius = 35 if self.camera_detected[idx] else 20
            cv2.circle(square_visual, position, radius, color, -1)
            cv2.putText(square_visual, f"Cam{idx+1}", (position[0] - 50, position[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return square_visual


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
