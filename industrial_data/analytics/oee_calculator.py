"""
Overall Equipment Effectiveness (OEE) Calculator

Calculates OEE = Availability × Performance × Quality
for manufacturing equipment monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta


class OEECalculator:
    """Calculate Overall Equipment Effectiveness metrics"""

    def calculate(
        self,
        scada_data: Dict[str, Any],
        work_orders: List[Any],
        time_window: timedelta = timedelta(hours=8)
    ) -> Dict[str, float]:
        """
        Calculate OEE metrics.

        Returns:
            {
                'oee': Overall OEE (0-100%),
                'availability': Availability (0-100%),
                'performance': Performance (0-100%),
                'quality': Quality (0-100%)
            }
        """
        availability = self._calculate_availability(scada_data, time_window)
        performance = self._calculate_performance(scada_data)
        quality = self._calculate_quality(work_orders)

        oee = (availability / 100.0) * (performance / 100.0) * (quality / 100.0) * 100.0

        return {
            'oee': oee,
            'availability': availability,
            'performance': performance,
            'quality': quality
        }

    def _calculate_availability(
        self,
        scada_data: Dict[str, Any],
        time_window: timedelta
    ) -> float:
        """
        Availability = (Operating Time / Planned Production Time) × 100

        Operating Time = Planned Production Time - Downtime
        """
        planned_time = time_window.total_seconds()

        # Extract downtime from SCADA tags
        downtime = 0.0
        for tag_key, tag_value in scada_data.items():
            if 'downtime' in tag_key.lower() or 'stopped' in tag_key.lower():
                if hasattr(tag_value, 'value'):
                    downtime += float(tag_value.value)

        operating_time = max(0, planned_time - downtime)
        availability = (operating_time / planned_time) * 100.0 if planned_time > 0 else 0.0

        return min(100.0, availability)

    def _calculate_performance(self, scada_data: Dict[str, Any]) -> float:
        """
        Performance = (Actual Output / Target Output) × 100

        Actual Output = Total Count
        Target Output = Operating Time × Ideal Cycle Time
        """
        actual_count = 0
        target_count = 1000  # Default target

        for tag_key, tag_value in scada_data.items():
            if 'production' in tag_key.lower() or 'count' in tag_key.lower():
                if hasattr(tag_value, 'value'):
                    actual_count = float(tag_value.value)
                    break

        performance = (actual_count / target_count) * 100.0 if target_count > 0 else 0.0
        return min(100.0, performance)

    def _calculate_quality(self, work_orders: List[Any]) -> float:
        """
        Quality = (Good Units / Total Units) × 100
        """
        if not work_orders:
            return 100.0

        total_units = 0
        defective_units = 0

        for wo in work_orders:
            if hasattr(wo, 'quantity'):
                total_units += wo.quantity
            if hasattr(wo, 'defects'):
                defective_units += wo.defects

        good_units = total_units - defective_units
        quality = (good_units / total_units) * 100.0 if total_units > 0 else 100.0

        return min(100.0, quality)
