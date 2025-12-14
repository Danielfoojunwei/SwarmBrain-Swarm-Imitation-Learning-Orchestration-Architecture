"""
Dynamic User Clustering for Federated Learning

Groups robots/sites by task, environment, and network quality to reduce
communication overhead and maintain accuracy.

Reference:
"Dynamic User Clustering and Resource Allocation for Federated Learning"
NTU research on efficient FL clustering
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import logging


class ClusteringMethod(Enum):
    """Clustering algorithms available."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    TASK_BASED = "task_based"
    ENVIRONMENT_BASED = "environment_based"
    NETWORK_BASED = "network_based"
    HYBRID = "hybrid"


@dataclass
class RobotProfile:
    """Profile of a robot for clustering."""
    robot_id: str
    task_type: str
    environment_features: np.ndarray  # e.g., [workspace_size, obstacle_density, lighting]
    network_quality: float  # 0.0 to 1.0
    bandwidth_mbps: float
    latency_ms: float
    model_performance: float  # Current accuracy/loss
    data_size: int  # Number of local training examples
    location: Optional[Tuple[float, float]] = None  # (lat, lon) if relevant


class DynamicClusterer:
    """
    Dynamic clustering of robots for federated learning.

    Adapts clusters based on task similarity, environment conditions,
    and network quality to optimize communication and convergence.
    """

    def __init__(
        self,
        method: ClusteringMethod = ClusteringMethod.HYBRID,
        num_clusters: Optional[int] = None,
        min_cluster_size: int = 2,
        max_cluster_size: Optional[int] = None
    ):
        """
        Initialize dynamic clusterer.

        Args:
            method: Clustering method to use
            num_clusters: Target number of clusters (auto-detected if None)
            min_cluster_size: Minimum robots per cluster
            max_cluster_size: Maximum robots per cluster (None = unlimited)
        """
        self.method = method
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.logger = logging.getLogger(__name__)

        self.scaler = StandardScaler()
        self.clusters: Dict[str, Set[str]] = {}  # cluster_id -> robot_ids
        self.robot_to_cluster: Dict[str, str] = {}  # robot_id -> cluster_id

    def cluster_robots(
        self,
        robot_profiles: List[RobotProfile]
    ) -> Dict[str, Set[str]]:
        """
        Cluster robots based on their profiles.

        Args:
            robot_profiles: List of robot profiles to cluster

        Returns:
            Dictionary mapping cluster IDs to sets of robot IDs
        """
        if len(robot_profiles) < self.min_cluster_size:
            self.logger.warning(
                f"Too few robots ({len(robot_profiles)}) for clustering, "
                "creating single cluster"
            )
            cluster_id = "cluster_0"
            self.clusters[cluster_id] = {p.robot_id for p in robot_profiles}
            for profile in robot_profiles:
                self.robot_to_cluster[profile.robot_id] = cluster_id
            return self.clusters

        # Extract features based on clustering method
        features = self._extract_features(robot_profiles)

        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        # Perform clustering
        if self.method == ClusteringMethod.KMEANS:
            labels = self._kmeans_clustering(features_normalized)
        elif self.method == ClusteringMethod.DBSCAN:
            labels = self._dbscan_clustering(features_normalized)
        elif self.method == ClusteringMethod.TASK_BASED:
            labels = self._task_based_clustering(robot_profiles)
        elif self.method == ClusteringMethod.ENVIRONMENT_BASED:
            labels = self._environment_clustering(features_normalized)
        elif self.method == ClusteringMethod.NETWORK_BASED:
            labels = self._network_clustering(features_normalized)
        else:  # HYBRID
            labels = self._hybrid_clustering(robot_profiles, features_normalized)

        # Build cluster assignments
        self.clusters.clear()
        self.robot_to_cluster.clear()

        for idx, (profile, label) in enumerate(zip(robot_profiles, labels)):
            cluster_id = f"cluster_{label}"

            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = set()

            self.clusters[cluster_id].add(profile.robot_id)
            self.robot_to_cluster[profile.robot_id] = cluster_id

        # Validate cluster sizes
        self._validate_clusters(robot_profiles)

        self.logger.info(
            f"Clustered {len(robot_profiles)} robots into {len(self.clusters)} clusters "
            f"using {self.method.value} method"
        )

        return self.clusters

    def _extract_features(self, profiles: List[RobotProfile]) -> np.ndarray:
        """Extract feature vectors from robot profiles."""
        features = []
        for profile in profiles:
            feature_vec = np.concatenate([
                profile.environment_features,
                [profile.network_quality],
                [profile.bandwidth_mbps / 1000.0],  # Normalize to Gbps
                [profile.latency_ms / 1000.0],  # Normalize to seconds
                [profile.model_performance],
                [np.log10(profile.data_size + 1)]  # Log scale for data size
            ])
            features.append(feature_vec)
        return np.array(features)

    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering."""
        n_clusters = self.num_clusters or self._estimate_num_clusters(features)
        n_clusters = min(n_clusters, len(features))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        return labels

    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering (density-based)."""
        dbscan = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
        labels = dbscan.fit_predict(features)

        # DBSCAN can produce -1 labels (noise), reassign to nearest cluster
        if -1 in labels:
            self.logger.warning("DBSCAN produced noise points, reassigning...")
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                # Assign to cluster with closest centroid
                labels[idx] = self._assign_to_nearest_cluster(
                    features[idx], features, labels
                )

        return labels

    def _task_based_clustering(self, profiles: List[RobotProfile]) -> np.ndarray:
        """Cluster by task type."""
        task_types = list(set(p.task_type for p in profiles))
        task_to_label = {task: i for i, task in enumerate(task_types)}

        labels = np.array([task_to_label[p.task_type] for p in profiles])
        return labels

    def _environment_clustering(self, features: np.ndarray) -> np.ndarray:
        """Cluster by environment features (first N features)."""
        # Assume first features are environment-related
        env_features = features[:, :3]  # Adjust based on actual feature layout
        return self._kmeans_clustering(env_features)

    def _network_clustering(self, features: np.ndarray) -> np.ndarray:
        """Cluster by network quality."""
        # Extract network-related features
        network_features = features[:, -4:-1]  # network_quality, bandwidth, latency
        return self._kmeans_clustering(network_features)

    def _hybrid_clustering(
        self,
        profiles: List[RobotProfile],
        features: np.ndarray
    ) -> np.ndarray:
        """
        Hybrid clustering considering task, environment, and network.

        Performs hierarchical clustering: first by task, then by network quality.
        """
        # First level: cluster by task type
        task_labels = self._task_based_clustering(profiles)

        # Second level: within each task cluster, sub-cluster by network quality
        final_labels = np.zeros_like(task_labels)
        next_label = 0

        for task_label in np.unique(task_labels):
            task_indices = np.where(task_labels == task_label)[0]

            if len(task_indices) >= self.min_cluster_size * 2:
                # Further divide by network quality
                task_features = features[task_indices]
                network_features = task_features[:, -4:-1]

                # Sub-cluster
                n_subclusters = max(2, len(task_indices) // self.min_cluster_size)
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                sublabels = kmeans.fit_predict(network_features)

                for sublabel in sublabels:
                    final_labels[task_indices[sublabels == sublabel]] = next_label
                    next_label += 1
            else:
                # Keep as single cluster
                final_labels[task_indices] = next_label
                next_label += 1

        return final_labels

    def _estimate_num_clusters(self, features: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method."""
        max_clusters = min(10, len(features) // self.min_cluster_size)

        if max_clusters < 2:
            return 1

        inertias = []
        k_range = range(2, max_clusters + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection: find point of maximum curvature
        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        elbow_idx = np.argmax(np.abs(np.diff(diffs))) + 1

        optimal_k = k_range[elbow_idx]
        self.logger.debug(f"Estimated optimal clusters: {optimal_k}")

        return optimal_k

    def _assign_to_nearest_cluster(
        self,
        point: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray
    ) -> int:
        """Assign a point to the nearest cluster centroid."""
        unique_labels = [l for l in np.unique(labels) if l != -1]

        if not unique_labels:
            return 0

        centroids = []
        for label in unique_labels:
            cluster_points = features[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)
        distances = np.linalg.norm(centroids - point, axis=1)
        nearest_label = unique_labels[np.argmin(distances)]

        return nearest_label

    def _validate_clusters(self, profiles: List[RobotProfile]):
        """Validate and fix cluster sizes."""
        # Check for clusters that are too small
        small_clusters = [
            cid for cid, robots in self.clusters.items()
            if len(robots) < self.min_cluster_size
        ]

        if small_clusters:
            self.logger.warning(
                f"Found {len(small_clusters)} clusters smaller than minimum size"
            )
            # Merge small clusters
            self._merge_small_clusters(profiles)

        # Check for clusters that are too large
        if self.max_cluster_size:
            large_clusters = [
                cid for cid, robots in self.clusters.items()
                if len(robots) > self.max_cluster_size
            ]

            if large_clusters:
                self.logger.warning(
                    f"Found {len(large_clusters)} clusters larger than maximum size"
                )
                # Split large clusters
                self._split_large_clusters(profiles)

    def _merge_small_clusters(self, profiles: List[RobotProfile]):
        """Merge clusters that are too small."""
        # Find small clusters
        small_cluster_ids = [
            cid for cid, robots in self.clusters.items()
            if len(robots) < self.min_cluster_size
        ]

        if len(small_cluster_ids) < 2:
            return

        # Merge all small clusters into one
        merged_robots = set()
        for cid in small_cluster_ids:
            merged_robots.update(self.clusters[cid])
            del self.clusters[cid]

        # Create new merged cluster
        new_cluster_id = f"cluster_merged_{len(self.clusters)}"
        self.clusters[new_cluster_id] = merged_robots

        # Update robot-to-cluster mapping
        for robot_id in merged_robots:
            self.robot_to_cluster[robot_id] = new_cluster_id

    def _split_large_clusters(self, profiles: List[RobotProfile]):
        """Split clusters that are too large."""
        large_cluster_ids = [
            cid for cid, robots in self.clusters.items()
            if self.max_cluster_size and len(robots) > self.max_cluster_size
        ]

        for cid in large_cluster_ids:
            robots = list(self.clusters[cid])
            robot_profiles = [p for p in profiles if p.robot_id in robots]

            # Re-cluster this subset
            num_splits = len(robots) // self.max_cluster_size + 1
            features = self._extract_features(robot_profiles)
            features_norm = self.scaler.transform(features)

            kmeans = KMeans(n_clusters=num_splits, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_norm)

            # Remove original cluster
            del self.clusters[cid]

            # Create new sub-clusters
            for split_idx in range(num_splits):
                new_cid = f"{cid}_split_{split_idx}"
                split_robots = {
                    robot_profiles[i].robot_id
                    for i, label in enumerate(labels)
                    if label == split_idx
                }
                self.clusters[new_cid] = split_robots

                for robot_id in split_robots:
                    self.robot_to_cluster[robot_id] = new_cid

    def get_cluster_for_robot(self, robot_id: str) -> Optional[str]:
        """Get cluster ID for a specific robot."""
        return self.robot_to_cluster.get(robot_id)

    def get_cluster_statistics(self) -> Dict[str, any]:
        """Get statistics about current clustering."""
        cluster_sizes = [len(robots) for robots in self.clusters.values()]

        return {
            'num_clusters': len(self.clusters),
            'total_robots': sum(cluster_sizes),
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'cluster_sizes': cluster_sizes
        }
