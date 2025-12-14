"""
Equipment Health Monitoring

Tracks equipment health metrics: MTBF, MTTR, vibration analysis, temperature trends.
Uses anomaly detection to identify potential failures.
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from scipy import signal


class EquipmentHealthMonitor:
    """Monitor equipment health with anomaly detection"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_history = []

    def get_health_metrics(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate equipment health metrics.

        Returns:
            {
                'mtbf': Mean Time Between Failures (hours),
                'mttr': Mean Time To Repair (hours),
                'vibration_anomaly_score': Anomaly score (0-1, higher=more anomalous),
                'temperature_trend': Trend direction ('increasing', 'stable', 'decreasing'),
                'health_score': Overall health (0-100)
            }
        """
        mtbf = self._calculate_mtbf()
        mttr = self._calculate_mttr()
        vibration_score = self._analyze_vibration(sensor_data)
        temp_trend = self._analyze_temperature_trend(sensor_data)
        health_score = self._calculate_health_score(mtbf, mttr, vibration_score)

        return {
            'mtbf': mtbf,
            'mttr': mttr,
            'vibration_anomaly_score': vibration_score,
            'temperature_trend': temp_trend,
            'health_score': health_score
        }

    def _calculate_mtbf(self) -> float:
        """Calculate Mean Time Between Failures"""
        if len(self.failure_history) < 2:
            return 1000.0  # Default: 1000 hours

        intervals = []
        for i in range(1, len(self.failure_history)):
            delta = self.failure_history[i] - self.failure_history[i-1]
            intervals.append(delta.total_seconds() / 3600.0)

        return np.mean(intervals) if intervals else 1000.0

    def _calculate_mttr(self) -> float:
        """Calculate Mean Time To Repair"""
        # Simplified: assume 2 hour average repair time
        return 2.0

    def _analyze_vibration(self, sensor_data: Dict[str, Any]) -> float:
        """Analyze vibration data for anomalies"""
        vibration_values = []

        for key, value in sensor_data.items():
            if 'vibration' in key.lower():
                if isinstance(value, dict) and 'payload' in value:
                    try:
                        vibration_values.append(float(value['payload']))
                    except (ValueError, TypeError):
                        pass

        if not vibration_values:
            return 0.0

        # Use simple threshold-based anomaly detection
        vibration_array = np.array(vibration_values).reshape(-1, 1)

        # Fit anomaly detector if we have enough samples
        if len(vibration_array) >= 10:
            self.anomaly_detector.fit(vibration_array)
            scores = self.anomaly_detector.score_samples(vibration_array)
            # Convert to 0-1 range (lower score = more anomalous)
            anomaly_score = 1.0 - (np.mean(scores) + 0.5)  # Normalize
            return max(0.0, min(1.0, anomaly_score))

        return 0.0

    def _analyze_temperature_trend(self, sensor_data: Dict[str, Any]) -> str:
        """Analyze temperature trends"""
        temp_values = []

        for key, value in sensor_data.items():
            if 'temperature' in key.lower() or 'temp' in key.lower():
                if isinstance(value, dict) and 'payload' in value:
                    try:
                        temp_values.append(float(value['payload']))
                    except (ValueError, TypeError):
                        pass

        if len(temp_values) < 5:
            return 'stable'

        # Linear regression to find trend
        x = np.arange(len(temp_values))
        y = np.array(temp_values)
        slope, _ = np.polyfit(x, y, 1)

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_health_score(
        self,
        mtbf: float,
        mttr: float,
        vibration_anomaly: float
    ) -> float:
        """Calculate overall equipment health score (0-100)"""
        # Higher MTBF = better health
        mtbf_score = min(100.0, (mtbf / 1000.0) * 100.0)

        # Lower MTTR = better health
        mttr_score = max(0.0, 100.0 - (mttr / 10.0) * 100.0)

        # Lower vibration anomaly = better health
        vibration_score = (1.0 - vibration_anomaly) * 100.0

        # Weighted average
        health = (
            0.4 * mtbf_score +
            0.3 * mttr_score +
            0.3 * vibration_score
        )

        return max(0.0, min(100.0, health))

    def record_failure(self, timestamp: datetime = None):
        """Record equipment failure for MTBF calculation"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        self.failure_history.append(timestamp)
