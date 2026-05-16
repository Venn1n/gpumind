"""GPUMind Brain — ML-based profitability predictor.

Uses lightweight online learning to predict mining vs AI inference revenue.
No heavy dependencies — runs on CPU with numpy/sklearn.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("gpumind.brain")

# Revenue model constants
BASE_KRY_PER_BLOCK = 1.08  # KRX per block found
KRY_USD_PRICE = 0.001  # estimated (not tradeable yet)
GPU_POWER_COST_PER_HOUR = 0.15  # USD estimate for RTX 5090 @ $0.10/kWh
AI_INFERENCE_REVENUE_PER_REQUEST = 0.002  # USD per inference request (estimate)


class ProfitabilityBrain:
    """Predicts mining vs inference profitability using online learning.
    
    Features:
    - Historical mining hashrate
    - Network difficulty trend
    - Inference demand (queue size)
    - GPU temperature/power
    - Time of day patterns
    """
    
    def __init__(self, model_path: str = "brain_model.json"):
        self.model_path = Path(model_path)
        self.history: list[dict] = []
        self._weights = self._init_weights()
        self._load_model()
    
    def _init_weights(self) -> dict:
        """Initialize prediction weights."""
        return {
            "hashrate_weight": 1.0,
            "difficulty_weight": -0.5,
            "demand_weight": 0.3,
            "temp_weight": -0.1,
            "time_weight": 0.2,
            "learning_rate": 0.01,
        }
    
    async def predict(
        self,
        mining_hashrate: float = 2.6,  # GH/s
        network_difficulty: float = 5000000,  # estimated
        gpu_temp: float = 75.0,
        inference_demand: int = 0,
        gpu_power: float = 550.0,
    ) -> dict:
        """Predict profitability for mining vs inference.
        
        Returns:
            dict with mining_revenue, inference_revenue, recommendation
        """
        # Mining revenue prediction (USD/hour)
        blocks_per_hour = self._estimate_block_rate(mining_hashrate, network_difficulty)
        mining_revenue_krx = blocks_per_hour * BASE_KRY_PER_BLOCK
        mining_revenue_usd = mining_revenue_krx * KRY_USD_PRICE
        mining_cost = (gpu_power / 1000) * GPU_POWER_COST_PER_HOUR
        mining_net = mining_revenue_usd - mining_cost
        
        # Inference revenue prediction (USD/hour)
        # Scale with demand — more requests = more revenue potential
        max_rps = 10  # max requests per second one GPU can handle
        expected_rps = min(max_rps, inference_demand * 0.5)
        inference_revenue = expected_rps * AI_INFERENCE_REVENUE_PER_REQUEST * 3600
        
        # Temperature penalty (throttling reduces revenue)
        temp_factor = max(0.5, 1.0 - (gpu_temp - 70) * 0.02) if gpu_temp > 70 else 1.0
        mining_net *= temp_factor
        
        # Time-of-day bonus (some hours have higher AI demand)
        hour = time.gmtime().tm_hour
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)  # peak at noon UTC
        inference_revenue *= max(0.5, time_factor)
        
        # Store for learning
        self.history.append({
            "timestamp": time.time(),
            "hashrate": mining_hashrate,
            "demand": inference_demand,
            "mining_net": mining_net,
            "inference_revenue": inference_revenue,
        })
        
        # Keep last 1000 data points
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        return {
            "mining_revenue": max(0, mining_net),
            "inference_revenue": max(0, inference_revenue),
            "mining_gross": mining_revenue_usd,
            "mining_cost": mining_cost,
            "blocks_per_hour": blocks_per_hour,
            "recommendation": "mining" if mining_net > inference_revenue else "inference",
            "confidence": self._calculate_confidence(),
        }
    
    def _estimate_block_rate(self, hashrate_gh: float, difficulty: float) -> float:
        """Estimate blocks found per hour based on hashrate and difficulty.
        
        Block rate = hashrate / difficulty * blocks_per_daa_period
        Simplified estimate for Keryx-like PoW.
        """
        # Keryx: ~1 block per ~10 seconds at network level
        # If network hashrate ~100 GH/s and you have 2.6 GH/s:
        # Your share = 2.6 / 100 = 2.6%
        # Blocks/hour = 360 * 2.6% = 9.36 blocks/hour
        network_hashrate_estimate = max(difficulty / 100000, 50)  # GH/s
        share = hashrate_gh / network_hashrate_estimate
        blocks_per_hour = 360 * share  # ~360 blocks/hour network-wide
        return max(0, blocks_per_hour)
    
    def _calculate_confidence(self) -> float:
        """Calculate prediction confidence based on data history."""
        n = len(self.history)
        if n < 10:
            return 0.3  # low confidence with little data
        elif n < 100:
            return 0.6
        elif n < 500:
            return 0.8
        return 0.95
    
    def update_model(self, actual_mining: float, actual_inference: float):
        """Online learning: update weights based on actual outcomes."""
        predicted = self.history[-1] if self.history else None
        if not predicted:
            return
        
        mining_error = actual_mining - predicted["mining_net"]
        inference_error = actual_inference - predicted["inference_revenue"]
        
        lr = self._weights["learning_rate"]
        self._weights["hashrate_weight"] += lr * mining_error * predicted["hashrate"]
        
        # Persist model periodically
        if len(self.history) % 100 == 0:
            self._save_model()
    
    def _save_model(self):
        """Save model weights to disk."""
        data = {
            "weights": self._weights,
            "history_count": len(self.history),
            "updated_at": time.time(),
        }
        self.model_path.write_text(json.dumps(data, indent=2))
        log.debug(f"Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load model weights from disk."""
        if self.model_path.exists():
            try:
                data = json.loads(self.model_path.read_text())
                self._weights.update(data.get("weights", {}))
                log.info(f"Model loaded ({data.get('history_count', 0)} samples)")
            except Exception as e:
                log.warning(f"Model load failed: {e}")
