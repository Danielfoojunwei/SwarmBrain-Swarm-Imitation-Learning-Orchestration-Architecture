"""
Unit tests for dynamic clustering module.
"""

import pytest
import numpy as np
from learning.clustering.dynamic_clustering import (
    DynamicClusterer,
    ClusteringMethod,
    RobotProfile,
    ReputationTier
)


@pytest.fixture
def sample_robot_profiles():
    """Create sample robot profiles for testing."""
    profiles = []

    # Factory robots (similar environment)
    for i in range(5):
        profiles.append(RobotProfile(
            robot_id=f'factory_robot_{i}',
            task_type='assembly',
            environment_features=np.array([10.0, 0.3, 0.8]),  # workspace, obstacles, lighting
            network_quality=0.9,
            bandwidth_mbps=100.0,
            latency_ms=10.0,
            model_performance=0.85,
            data_size=1000
        ))

    # Warehouse robots (different environment)
    for i in range(5):
        profiles.append(RobotProfile(
            robot_id=f'warehouse_robot_{i}',
            task_type='picking',
            environment_features=np.array([50.0, 0.1, 0.6]),
            network_quality=0.7,
            bandwidth_mbps=50.0,
            latency_ms=50.0,
            model_performance=0.75,
            data_size=500
        ))

    return profiles


class TestDynamicClusterer:
    """Test suite for DynamicClusterer."""

    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = DynamicClusterer(
            method=ClusteringMethod.KMEANS,
            num_clusters=2
        )

        assert clusterer.method == ClusteringMethod.KMEANS
        assert clusterer.num_clusters == 2
        assert len(clusterer.clusters) == 0
        assert len(clusterer.robot_to_cluster) == 0

    def test_kmeans_clustering(self, sample_robot_profiles):
        """Test K-means clustering."""
        clusterer = DynamicClusterer(
            method=ClusteringMethod.KMEANS,
            num_clusters=2,
            min_cluster_size=2
        )

        clusters = clusterer.cluster_robots(sample_robot_profiles)

        # Should create 2 clusters
        assert len(clusters) >= 2

        # All robots should be assigned
        total_robots = sum(len(robots) for robots in clusters.values())
        assert total_robots == len(sample_robot_profiles)

        # Each cluster should have at least min_cluster_size robots
        for robots in clusters.values():
            assert len(robots) >= clusterer.min_cluster_size

    def test_task_based_clustering(self, sample_robot_profiles):
        """Test task-based clustering."""
        clusterer = DynamicClusterer(
            method=ClusteringMethod.TASK_BASED,
            min_cluster_size=2
        )

        clusters = clusterer.cluster_robots(sample_robot_profiles)

        # Should create 2 clusters (assembly and picking)
        assert len(clusters) == 2

        # Verify robots are grouped by task
        for cluster_id, robot_ids in clusters.items():
            tasks = set()
            for profile in sample_robot_profiles:
                if profile.robot_id in robot_ids:
                    tasks.add(profile.task_type)

            # All robots in cluster should have same task
            assert len(tasks) == 1

    def test_hybrid_clustering(self, sample_robot_profiles):
        """Test hybrid clustering (task + network)."""
        clusterer = DynamicClusterer(
            method=ClusteringMethod.HYBRID,
            min_cluster_size=2
        )

        clusters = clusterer.cluster_robots(sample_robot_profiles)

        # Should create clusters
        assert len(clusters) >= 2

        # All robots should be assigned
        total_robots = sum(len(robots) for robots in clusters.values())
        assert total_robots == len(sample_robot_profiles)

    def test_get_cluster_for_robot(self, sample_robot_profiles):
        """Test retrieving cluster for a specific robot."""
        clusterer = DynamicClusterer(method=ClusteringMethod.KMEANS, num_clusters=2)
        clusterer.cluster_robots(sample_robot_profiles)

        robot_id = sample_robot_profiles[0].robot_id
        cluster_id = clusterer.get_cluster_for_robot(robot_id)

        assert cluster_id is not None
        assert robot_id in clusterer.clusters[cluster_id]

    def test_cluster_statistics(self, sample_robot_profiles):
        """Test cluster statistics calculation."""
        clusterer = DynamicClusterer(method=ClusteringMethod.KMEANS, num_clusters=2)
        clusterer.cluster_robots(sample_robot_profiles)

        stats = clusterer.get_cluster_statistics()

        assert stats['total_robots'] == len(sample_robot_profiles)
        assert stats['num_clusters'] >= 2
        assert stats['avg_cluster_size'] > 0
        assert stats['min_cluster_size'] >= clusterer.min_cluster_size

    def test_min_cluster_size_validation(self):
        """Test that clusters respect minimum size."""
        # Create robots that might naturally cluster into small groups
        profiles = [
            RobotProfile(
                robot_id=f'robot_{i}',
                task_type='task_a' if i < 2 else 'task_b',
                environment_features=np.random.randn(3),
                network_quality=0.8,
                bandwidth_mbps=100.0,
                latency_ms=10.0,
                model_performance=0.8,
                data_size=1000
            )
            for i in range(10)
        ]

        clusterer = DynamicClusterer(
            method=ClusteringMethod.KMEANS,
            num_clusters=3,
            min_cluster_size=3
        )

        clusters = clusterer.cluster_robots(profiles)

        # All clusters should meet minimum size
        for cluster_id, robot_ids in clusters.items():
            assert len(robot_ids) >= clusterer.min_cluster_size

    def test_empty_profiles(self):
        """Test handling of empty robot profiles."""
        clusterer = DynamicClusterer(method=ClusteringMethod.KMEANS)

        clusters = clusterer.cluster_robots([])

        assert len(clusters) == 0

    def test_single_robot(self):
        """Test clustering with a single robot."""
        profile = RobotProfile(
            robot_id='solo_robot',
            task_type='solo_task',
            environment_features=np.array([1.0, 1.0, 1.0]),
            network_quality=0.9,
            bandwidth_mbps=100.0,
            latency_ms=10.0,
            model_performance=0.9,
            data_size=1000
        )

        clusterer = DynamicClusterer(
            method=ClusteringMethod.KMEANS,
            min_cluster_size=1
        )

        clusters = clusterer.cluster_robots([profile])

        assert len(clusters) == 1
        assert 'solo_robot' in list(clusters.values())[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
