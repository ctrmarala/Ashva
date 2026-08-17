"""
Ashva Smart Order Router (SOR) & Execution Slicer
Implements TWAP, VWAP, and Iceberg execution algorithms to minimize market impact and slippage.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np

from src.core.events import OrderEvent, OrderSide, OrderType, ProductType


class SmartOrderRouter:
    """
    Slices large parent orders into algorithmic child slices.
    """

    @staticmethod
    def slice_twap(
        parent_order: OrderEvent,
        total_duration_minutes: int = 30,
        num_slices: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Slices parent order into equal Time-Weighted Average Price (TWAP) micro-child orders.
        """
        if num_slices <= 1 or parent_order.quantity <= 1:
            return [{
                "slice_id": f"{parent_order.order_id}_1",
                "quantity": parent_order.quantity,
                "scheduled_time": parent_order.timestamp,
                "order_type": parent_order.order_type,
            }]

        base_qty = parent_order.quantity // num_slices
        remainder = parent_order.quantity % num_slices
        interval_seconds = (total_duration_minutes * 60) // num_slices

        slices = []
        for i in range(num_slices):
            qty = base_qty + (1 if i < remainder else 0)
            if qty <= 0:
                continue
            sched_time = parent_order.timestamp + timedelta(seconds=i * interval_seconds)
            slices.append({
                "slice_id": f"{parent_order.order_id}_{i+1}",
                "quantity": qty,
                "scheduled_time": sched_time,
                "order_type": parent_order.order_type,
            })
        return slices

    @staticmethod
    def slice_iceberg(
        parent_order: OrderEvent,
        visible_clip_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Slices order into visible iceberg clips to conceal full order book footprint.
        """
        total_qty = parent_order.quantity
        if total_qty <= visible_clip_size:
            return [{
                "clip_id": f"{parent_order.order_id}_clip_1",
                "quantity": total_qty,
                "is_visible": True,
            }]

        clips = []
        remaining = total_qty
        clip_idx = 1

        while remaining > 0:
            qty = min(remaining, visible_clip_size)
            clips.append({
                "clip_id": f"{parent_order.order_id}_clip_{clip_idx}",
                "quantity": qty,
                "is_visible": True,
            })
            remaining -= qty
            clip_idx += 1

        return clips
