from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    A class for assigning players to teams based on jersey colors using K-means clustering.

    This class analyzes player jersey colors from video frames and groups players into
    two teams based on color similarity. It uses computer vision techniques to extract
    dominant colors from player bounding boxes and machine learning clustering to
    distinguish between teams.

    Attributes:
        team_colors (Dict[int, np.ndarray]): Maps team IDs to their representative RGB colors
        player_team_dict (Dict[int, int]): Cache mapping player IDs to their assigned team IDs
        kmeans (KMeans): Trained K-means model for team color classification
    """

    def __init__(self) -> None:
        """Initialize the TeamAssigner with empty team colors and player assignments."""
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Create and train a K-means clustering model on image pixels.

        Reshapes the input image to a 2D array of pixels and applies K-means clustering
        with 2 clusters to separate foreground (player jersey) from background colors.

        Args:
            image (np.ndarray): Input image array with shape (height, width, 3)

        Returns:
            KMeans: Trained K-means model with 2 clusters

        Note:
            Uses k-means++ initialization for better cluster center selection
        """
        # Reshape the image to 2D array (pixels, RGB_channels)
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def _get_player_color(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract the dominant jersey color of a player from their bounding box.

        Crops the player region from the frame, focuses on the top half (where jersey
        is most visible), and uses clustering to separate jersey color from background.
        Uses corner pixels to identify background and extract the player's jersey color.

        Args:
            frame (np.ndarray): Video frame containing the player
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            np.ndarray: RGB color array representing the player's jersey color
        """
        # Extract player region from frame using bounding box
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Focus on top half where jersey is most visible
        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get clustering model for color separation
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to match image dimensions
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Identify background cluster using corner pixels (assumes corners are background)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Extract the dominant player jersey color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(
        self, frame: np.ndarray, player_detections: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Analyze all players in a frame and assign team colors based on jersey similarity.

        Extracts colors from all detected players and uses K-means clustering to group
        them into two teams. This method should be called once per video to establish
        team color assignments.

        Args:
            frame (np.ndarray): Video frame containing all players
            player_detections (Dict[int, Dict[str, Any]]): Dictionary mapping player IDs
                to their detection data including 'bbox' key

        Side Effects:
            - Sets self.kmeans with trained clustering model
            - Populates self.team_colors with team color assignments
        """
        player_colors = []

        # Extract colors from all detected players
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self._get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        # Store the trained model and team colors
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(
        self, frame: np.ndarray, player_bbox: List[float], player_id: int
    ) -> int:
        """
        Determine which team a specific player belongs to based on their jersey color.

        Uses the previously trained team color model to classify a player's jersey color
        and assign them to the appropriate team. Implements caching to avoid recomputation
        for the same player across frames.

        Args:
            frame (np.ndarray): Video frame containing the player
            player_bbox (List[float]): Player's bounding box coordinates [x1, y1, x2, y2]
            player_id (int): Unique identifier for the player

        Returns:
            int: Team ID (1 or 2) indicating which team the player belongs to

        Raises:
            AttributeError: If assign_team_color() hasn't been called first

        Note:
            - Returns cached result if player has been classified before
            - Includes special case handling for player ID 91 (forced to team 1)
            - Team IDs are 1-indexed (1 or 2, not 0 or 1)
        """
        # Return cached result if player already classified
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Check if model is trained
        if self.kmeans is None:
            raise AttributeError(
                "Team colors must be assigned first using assign_team_color()"
            )

        # Extract player's current jersey color
        player_color = self._get_player_color(frame, player_bbox)

        # Predict team based on color similarity to team clusters
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Convert from 0-indexed to 1-indexed

        # Special case: Force player 91 to team 1 (manual override)
        if player_id == 91:
            team_id = 1

        # Cache the result for future frames
        self.player_team_dict[player_id] = team_id

        return team_id
