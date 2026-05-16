"""GPUMind Monitor — Metrics collection and reporting."""

import time
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import GPUState

log = logging.getLogger("gpumind.monitor")


class MetricsCollector:
    """Collects and aggregates metrics over time."""
    
    def __init__(self, window_size: int = 360):
        """window_size: number of data points to keep (default: 1 hour at 10s intervals)"""
        self.window_size = window_size
        self._data: dict[int, list[dict]] = defaultdict(list)
        self._switches: list[dict] = []
        self._start_time = time.time()
    
    def record(self, gpu_id: int, state: "GPUState", prediction: dict = None):
        """Record a data point."""
        point = {
            "ts": time.time(),
            "mode": state.mode.value,
            "hashrate": state.mining_hashrate,
            "temperature": state.temperature,
            "power": state.power_watts,
            "utilization": state.utilization,
        }
        if prediction:
            point["mining_revenue"] = prediction.get("mining_revenue", 0)
            point["inference_revenue"] = prediction.get("inference_revenue", 0)
        
        self._data[gpu_id].append(point)
        
        # Trim to window
        if len(self._data[gpu_id]) > self.window_size:
            self._data[gpu_id] = self._data[gpu_id][-self.window_size:]
    
    def record_switch(self, gpu_id: int, from_mode: str, to_mode: str):
        """Record a mode switch event."""
        self._switches.append({
            "ts": time.time(),
            "gpu": gpu_id,
            "from": from_mode,
            "to": to_mode,
        })
        # Keep last 100 switches
        if len(self._switches) > 100:
            self._switches = self._switches[-100:]
    
    def summary(self) -> dict:
        """Get metrics summary."""
        uptime = time.time() - self._start_time
        gpu_summary = {}
        
        for gpu_id, points in self._data.items():
            if not points:
                continue
            
            hr_vals = [p["hashrate"] for p in points if p["hashrate"] > 0]
            temp_vals = [p["temperature"] for p in points if p["temperature"] > 0]
            power_vals = [p["power"] for p in points if p["power"] > 0]
            
            gpu_summary[gpu_id] = {
                "avg_hashrate": sum(hr_vals) / len(hr_vals) if hr_vals else 0,
                "max_hashrate": max(hr_vals) if hr_vals else 0,
                "avg_temperature": sum(temp_vals) / len(temp_vals) if temp_vals else 0,
                "max_temperature": max(temp_vals) if temp_vals else 0,
                "avg_power": sum(power_vals) / len(power_vals) if power_vals else 0,
                "samples": len(points),
            }
        
        return {
            "uptime_hours": uptime / 3600,
            "total_switches": len(self._switches),
            "recent_switches": self._switches[-5:],
            "gpus": gpu_summary,
        }
    
    def export_prometheus(self) -> dict:
        """Export metrics in Prometheus-like format."""
        metrics = {}
        summary = self.summary()
        
        for gpu_id, gpu_data in summary.get("gpus", {}).items():
            prefix = f"gpumind_gpu{gpu_id}"
            metrics[f"{prefix}_hashrate_avg"] = gpu_data["avg_hashrate"]
            metrics[f"{prefix}_hashrate_max"] = gpu_data["max_hashrate"]
            metrics[f"{prefix}_temp_avg"] = gpu_data["avg_temperature"]
            metrics[f"{prefix}_temp_max"] = gpu_data["max_temperature"]
            metrics[f"{prefix}_power_avg"] = gpu_data["avg_power"]
        
        metrics["gpumind_uptime_hours"] = summary["uptime_hours"]
        metrics["gpumind_total_switches"] = summary["total_switches"]
        
        return metrics
    
    def estimate_earnings(self, gpu_id: int, hours: float = 24) -> dict:
        """Estimate earnings over a period."""
        points = self._data.get(gpu_id, [])
        if not points:
            return {"mining_krx": 0, "inference_usd": 0}
        
        mining_points = [p for p in points if p.get("mining_revenue")]
        inference_points = [p for p in points if p.get("inference_revenue")]
        
        avg_mining = sum(p["mining_revenue"] for p in mining_points) / len(mining_points) if mining_points else 0
        avg_inference = sum(p["inference_revenue"] for p in inference_points) / len(inference_points) if inference_points else 0
        
        return {
            "mining_usd": avg_mining * hours,
            "inference_usd": avg_inference * hours,
            "total_usd": (avg_mining + avg_inference) * hours,
            "period_hours": hours,
        }
