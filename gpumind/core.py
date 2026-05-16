"""GPUMind Core — Main daemon that orchestrates mining/inference switching."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml

from .brain import ProfitabilityBrain
from .gpu_manager import GPUManager
from .monitor import MetricsCollector
from .api import APIServer

log = logging.getLogger("gpumind")


class Mode(str, Enum):
    MINING = "mining"
    INFERENCE = "inference"
    HYBRID = "hybrid"
    BURST = "burst"
    IDLE = "idle"


@dataclass
class GPUState:
    gpu_id: int
    mode: Mode = Mode.IDLE
    mining_hashrate: float = 0.0  # GH/s
    inference_rps: float = 0.0    # requests per second
    temperature: float = 0.0
    power_watts: float = 0.0
    utilization: float = 0.0
    last_switch: float = 0.0


@dataclass
class GPUMindConfig:
    wallet: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_port: int = 8500
    check_interval: int = 300  # seconds
    min_mode_duration: int = 600  # minimum seconds before switch
    mining_priority: float = 0.6  # 0-1, higher = prefer mining
    inference_queue_threshold: int = 5  # requests before switching to inference
    gpu_ids: list = field(default_factory=lambda: [0])
    
    @classmethod
    def from_yaml(cls, path: str) -> "GPUMindConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GPUMind:
    """Main daemon — monitors, predicts, and switches GPU modes."""
    
    def __init__(self, config: GPUMindConfig):
        self.config = config
        self.gpu = GPUManager(config.gpu_ids)
        self.brain = ProfitabilityBrain()
        self.monitor = MetricsCollector()
        self.api = APIServer(self, port=config.api_port)
        self.states: dict[int, GPUState] = {
            gid: GPUState(gpu_id=gid) for gid in config.gpu_ids
        }
        self._running = False
        self._inference_queue: asyncio.Queue = asyncio.Queue()
    
    async def start(self):
        """Start all components."""
        log.info("🧠 GPUMind starting...")
        self._running = True
        
        # Init GPU discovery
        gpus = self.gpu.discover()
        log.info(f"Found {len(gpus)} GPU(s): {[g['name'] for g in gpus]}")
        
        # Start API server
        asyncio.create_task(self.api.start())
        log.info(f"API server on :{self.config.api_port}")
        
        # Main loop
        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                log.error(f"Tick error: {e}")
                await asyncio.sleep(30)
    
    async def stop(self):
        """Graceful shutdown."""
        log.info("GPUMind stopping...")
        self._running = False
        for state in self.states.values():
            await self._switch_mode(state, Mode.IDLE)
    
    async def _tick(self):
        """Single decision cycle for each GPU."""
        for gid, state in self.states.items():
            # Gather current metrics
            metrics = self.gpu.get_metrics(gid)
            state.temperature = metrics.get("temperature", 0)
            state.power_watts = metrics.get("power", 0)
            state.utilization = metrics.get("utilization", 0)
            
            # Get profitability prediction
            prediction = await self.brain.predict(
                mining_hashrate=state.mining_hashrate,
                gpu_temp=state.temperature,
                inference_demand=self._inference_queue.qsize(),
            )
            
            # Decision logic
            target_mode = self._decide_mode(state, prediction)
            
            # Check minimum duration
            elapsed = time.time() - state.last_switch
            if target_mode != state.mode and elapsed > self.config.min_mode_duration:
                await self._switch_mode(state, target_mode)
            
            # Record metrics
            self.monitor.record(gid, state, prediction)
    
    def _decide_mode(self, state: GPUState, prediction: dict) -> Mode:
        """Decide optimal mode based on prediction and current state."""
        mining_score = prediction["mining_revenue"] * self.config.mining_priority
        inference_score = prediction["inference_revenue"] * (1 - self.config.mining_priority)
        queue_size = self._inference_queue.qsize()
        
        # Burst mode: instant interrupt for high-value inference
        if queue_size >= self.config.inference_queue_threshold * 3:
            return Mode.BURST
        
        # Inference demand override
        if queue_size >= self.config.inference_queue_threshold:
            return Mode.INFERENCE
        
        # Profitability comparison
        if inference_score > mining_score * 1.2:  # 20% threshold
            return Mode.INFERENCE
        
        if mining_score > inference_score * 1.2:
            return Mode.MINING
        
        # Close scores → hybrid
        return Mode.HYBRID
    
    async def _switch_mode(self, state: GPUState, target: Mode):
        """Switch GPU to target mode."""
        old_mode = state.mode
        log.info(f"GPU {state.gpu_id}: {old_mode} → {target}")
        
        # Stop current mode
        if old_mode == Mode.MINING:
            self.gpu.stop_mining(state.gpu_id)
        elif old_mode == Mode.INFERENCE:
            self.gpu.stop_inference(state.gpu_id)
        
        # Start new mode
        if target == Mode.MINING:
            hr = self.gpu.start_mining(state.gpu_id, self.config.wallet)
            state.mining_hashrate = hr
        elif target == Mode.INFERENCE:
            self.gpu.start_inference(state.gpu_id)
        elif target == Mode.HYBRID:
            hr = self.gpu.start_mining(state.gpu_id, self.config.wallet, power_limit=70)
            self.gpu.start_inference(state.gpu_id)
            state.mining_hashrate = hr * 0.7
        elif target == Mode.BURST:
            self.gpu.stop_mining(state.gpu_id)
            self.gpu.start_inference(state.gpu_id)
        
        state.mode = target
        state.last_switch = time.time()
        
        # Telegram notification
        await self._notify_mode_switch(state, old_mode, target)
    
    async def _notify_mode_switch(self, state: GPUState, old: Mode, new: Mode):
        """Send Telegram alert on mode switch."""
        if not self.config.telegram_bot_token:
            return
        
        import urllib.request
        msg = (
            f"🧠 **GPUMind Mode Switch**\n"
            f"🖥️ GPU {state.gpu_id}\n"
            f"📊 {old.value} → {new.value}\n"
            f"🌡️ {state.temperature:.0f}°C | ⚡ {state.power_watts:.0f}W"
        )
        url = (
            f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            f"?chat_id={self.config.telegram_chat_id}"
            f"&parse_mode=Markdown"
            f"&text={urllib.parse.quote(msg)}"
        )
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:
            pass
    
    def get_status(self) -> dict:
        """Get current status for API/dashboard."""
        return {
            "version": "0.1.0",
            "gpus": [
                {
                    "id": s.gpu_id,
                    "mode": s.mode.value,
                    "hashrate": s.mining_hashrate,
                    "temperature": s.temperature,
                    "power": s.power_watts,
                    "utilization": s.utilization,
                }
                for s in self.states.values()
            ],
            "inference_queue": self._inference_queue.qsize(),
            "metrics": self.monitor.summary(),
        }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPUMind — AI GPU Resource Manager")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    config = GPUMindConfig.from_yaml(args.config)
    daemon = GPUMind(config)
    
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        asyncio.run(daemon.stop())


if __name__ == "__main__":
    main()
