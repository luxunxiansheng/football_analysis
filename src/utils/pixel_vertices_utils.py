import cv2
import numpy as np
from typing import List, Optional


class PixelVertexSelector:
    """
    Interactive tool to manually select pixel vertices for court corner points.
    """

    def __init__(self):
        self.points: List[List[int]] = []
        self.image: np.ndarray = np.array([])

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse click events to record corner points."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            print(f"Point {len(self.points)}: [{x}, {y}]")
            if self.image is not None and self.image.size > 0:
                cv2.circle(self.image, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(
                    self.image,
                    f"{len(self.points)}",
                    (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Select Court Corners", self.image)
            if len(self.points) == 4:
                print("\n" + "=" * 50)
                print("All 4 points selected!")
                print("Copy this line to your ViewTransformer:")
                print(f"self.pixel_vertices = np.array({self.points})")
                print("=" * 50)

    def select_vertices(self, image_path: str) -> Optional[List[List[int]]]:
        """
        Load image and allow user to select 4 corner points.

        Args:
            image_path (str): Path to the image/video frame

        Returns:
            Optional[List[List[int]]]: List of 4 selected points or None if not completed
        """
        self.image = cv2.imread(image_path)
        if self.image is None or self.image.size == 0:
            print(f"Failed to load image: {image_path}")
            return None
        cv2.imshow("Select Court Corners", self.image)
        cv2.setMouseCallback("Select Court Corners", self.mouse_callback)
        while len(self.points) < 4:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        return self.points if len(self.points) == 4 else None


def extract_frame_from_video(video_path: str, frame_number: int = 100) -> Optional[str]:
    """
    Extract a frame from video for vertex selection.

    Args:
        video_path (str): Path to video file
        frame_number (int): Frame to extract (default: 100)

    Returns:
        Optional[str]: Path to extracted frame or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        output_path = f"/workspaces/football_analysis/frame_{frame_number}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} extracted to: {output_path}")
        cap.release()
        return output_path
    else:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return None


# Usage example
if __name__ == "__main__":
    # Option 1: Use existing image
    image_path = "/workspaces/football_analysis/sample_frame.jpg"

    # Option 2: Extract frame from video (uncomment if needed)
    # video_path = "/workspaces/football_analysis/input_videos/your_video.mp4"
    # image_path = extract_frame_from_video(video_path, frame_number=100)

    if image_path:
        selector = PixelVertexSelector()
        vertices = selector.select_vertices(image_path)

        if vertices:
            print(f"\nSuccess! Use these coordinates in ViewTransformer:")
            print(f"self.pixel_vertices = np.array({vertices})")
        else:
            print("Failed to select all 4 vertices.")
