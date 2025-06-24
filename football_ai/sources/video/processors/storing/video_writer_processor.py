from football_ai.utils import cv2, np, create_progress_bar
from football_ai.domain import Video
from football_ai.domain.interfaces import Processor


class VideoWriterProcessor(Processor):
    """
    Writes the frames in VideoData to a video file using OpenCV.
    Assumes each frame's raw_frame is a numpy array (BGR).
    """

    def __init__(self, output_path: str, codec: str = "mp4v"):
        self.output_path = output_path
        self.codec = codec

    def process(self, video_data: Video) -> Video:
        if not video_data.frames:
            raise ValueError("No frames to write in VideoData.")
        height, width = video_data.resolution[1], video_data.resolution[0]
        fourcc = cv2.VideoWriter.fourcc(*self.codec)
        out = cv2.VideoWriter(
            self.output_path, fourcc, video_data.frame_rate, (width, height)
        )

        # Use progress bar for frame writing
        progress_bar = create_progress_bar(
            iterable=video_data.frames, desc="Writing video frames", unit="frames"
        )

        for frame_data in progress_bar:
            frame = frame_data.raw_frame
            if frame is not None:
                # Ensure frame is uint8 and 3-channel
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                out.write(frame)

        progress_bar.close()
        out.release()
        return video_data
