from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
from typing import List, Dict, Any, Optional, Tuple, Union
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    """
    A comprehensive object tracking system for football analysis using YOLO and ByteTrack.
    
    This class provides functionality to detect and track players, referees, and the ball
    in football video footage. It includes methods for interpolation, annotation drawing,
    and team ball control visualization.
    
    Attributes:
        model (YOLO): YOLO object detection model
        tracker (ByteTrack): ByteTrack tracker for object tracking
    """
    
    def __init__(self, model_path: str) -> None:
        """
        Initialize the Tracker with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        self.model: YOLO = YOLO(model_path) 
        self.tracker: sv.ByteTrack = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Dict[str, List[Dict[int, Dict[str, Any]]]]) -> None:
        """
        Add position information to tracking data for all tracked objects.
        
        For balls, uses the center of the bounding box as position.
        For players and referees, uses foot position for more accurate ground positioning.
        
        Args:
            tracks (Dict[str, List[Dict[int, Dict[str, Any]]]]): Tracking data structure containing:
                - 'players': List of frame dictionaries with player tracking info
                - 'referees': List of frame dictionaries with referee tracking info  
                - 'ball': List of frame dictionaries with ball tracking info
        
        Note:
            Modifies the tracks dictionary in-place by adding 'position' key to each track.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, Dict[str, List[float]]]]) -> List[Dict[int, Dict[str, List[float]]]]:
        """
        Interpolate missing ball positions to ensure smooth ball tracking.
        
        Uses pandas interpolation to fill gaps in ball detection, which commonly
        occur when the ball is occluded or moves too fast.
        
        Args:
            ball_positions (List[Dict[int, Dict[str, List[float]]]]): List of dictionaries containing ball tracking data
                Format: [{1: {"bbox": [x1, y1, x2, y2]}}, ...]
        
        Returns:
            List[Dict[int, Dict[str, List[float]]]]: Interpolated ball positions in the same format as input
        
        Example:
            >>> ball_pos = [{1: {"bbox": [100, 100, 120, 120]}}, {}, {1: {"bbox": [130, 130, 150, 150]}}]
            >>> interpolated = tracker.interpolate_ball_positions(ball_pos)
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames: List[np.ndarray]) -> List[Any]:
        """
        Perform object detection on a list of video frames using batch processing.
        
        Processes frames in batches to optimize GPU memory usage and inference speed.
        
        Args:
            frames (List[np.ndarray]): List of video frames (numpy arrays)
        
        Returns:
            List[Any]: List of YOLO detection results, one per frame
        
        Note:
            Uses batch_size=20 and confidence threshold=0.1 for detection
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames: List[np.ndarray], read_from_stub: bool = False, stub_path: Optional[str] = None) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """
        Generate comprehensive tracking data for all objects in video frames.
        
        Detects and tracks players, referees, and ball across all frames. Supports
        caching to/from pickle files for faster subsequent processing.
        
        Args:
            frames (List[np.ndarray]): List of video frames to process
            read_from_stub (bool): Whether to load from cached file
            stub_path (Optional[str]): Path to cache file for saving/loading
        
        Returns:
            Dict[str, List[Dict[int, Dict[str, Any]]]]: Comprehensive tracking data with structure:
                {
                    "players": [frame_dict, ...],    # Player tracking per frame
                    "referees": [frame_dict, ...],   # Referee tracking per frame  
                    "ball": [frame_dict, ...]        # Ball tracking per frame
                }
                where frame_dict = {track_id: {"bbox": [x1,y1,x2,y2]}}
        
        Note:
            - Converts goalkeepers to player class automatically
            - Ball uses fixed track_id=1, other objects get dynamic IDs
            - Saves results to stub_path if provided
        """
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)    

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int], track_id: Optional[int] = None) -> np.ndarray:
        """
        Draw an ellipse around tracked objects (players/referees) at foot level.
        
        Creates an ellipse at the bottom of the bounding box to represent the player's
        ground position, with an optional track ID label.
        
        Args:
            frame (np.ndarray): Video frame to draw on
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
            color (Tuple[int, int, int]): BGR color for the ellipse (B, G, R)
            track_id (Optional[int]): Track ID to display in label
        
        Returns:
            np.ndarray: Frame with ellipse and label drawn
        
        Note:
            - Ellipse is positioned at the bottom center of bbox
            - Rectangle label adjusts width for track_id > 99
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw a triangle marker above tracked objects (typically for ball or special indicators).
        
        Creates a triangle pointing downward at the top of the bounding box, commonly
        used to mark the ball or indicate which player has possession.
        
        Args:
            frame (np.ndarray): Video frame to draw on
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
            color (Tuple[int, int, int]): BGR color for the triangle (B, G, R)
        
        Returns:
            np.ndarray: Frame with triangle drawn
        
        Note:
            Triangle is positioned at top center of bbox with black outline
        """
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray) -> np.ndarray:
        """
        Draw team ball control statistics overlay on the video frame.
        
        Creates a semi-transparent panel showing the percentage of time each team
        has controlled the ball up to the current frame.
        
        Args:
            frame (np.ndarray): Video frame to draw on
            frame_num (int): Current frame number
            team_ball_control (np.ndarray): Array indicating which team (1 or 2) 
                                             controls ball in each frame
        
        Returns:
            np.ndarray: Frame with ball control statistics overlay
        
        Note:
            - Overlay positioned at bottom-right of frame
            - Shows cumulative percentages from start to current frame
            - Uses white semi-transparent background
        """
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames: List[np.ndarray], tracks: Dict[str, List[Dict[int, Dict[str, Any]]]], team_ball_control: np.ndarray) -> List[np.ndarray]:
        """
        Apply all visual annotations to video frames for comprehensive football analysis.
        
        Processes each frame to add tracking visualizations including player ellipses,
        referee markers, ball indicators, possession triangles, and team statistics.
        
        Args:
            video_frames (List[np.ndarray]): List of video frames to annotate
            tracks (Dict[str, List[Dict[int, Dict[str, Any]]]]): Tracking data from get_object_tracks()
            team_ball_control (np.ndarray): Team ball control data per frame
        
        Returns:
            List[np.ndarray]: List of annotated video frames ready for output
        
        Visual Elements Added:
            - Colored ellipses for players (team colors)
            - Yellow ellipses for referees  
            - Green triangle for ball
            - Red triangle for player with possession
            - Ball control statistics overlay
        
        Note:
            Expects tracks to contain 'team_color' and 'has_ball' keys for players
        """
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames