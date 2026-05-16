"""GPUMind GPU Manager — Handles mining/inference process lifecycle."""

import subprocess
import logging
import signal
import os
from typing import Optional

log = logging.getLogger("gpumind.gpu")


class GPUManager:
    """Manages GPU processes — mining and inference servers."""
    
    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self._mining_procs: dict[int, subprocess.Popen] = {}
        self._inference_procs: dict[int, subprocess.Popen] = {}
    
    def discover(self) -> list[dict]:
        """Discover available GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({
                        "id": int(parts[0]),
                        "name": parts[1],
                        "memory_total": parts[2],
                    })
            return gpus
        except Exception as e:
            log.error(f"GPU discovery failed: {e}")
            return []
    
    def get_metrics(self, gpu_id: int) -> dict:
        """Get current GPU metrics."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                    "-i", str(gpu_id)
                ],
                capture_output=True, text=True, timeout=5
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "temperature": float(parts[0].replace(" C", "")) if parts[0] != "[N/A]" else 0,
                "power": float(parts[1].replace(" W", "")) if "[N/A]" not in parts[1] else 0,
                "utilization": float(parts[2].replace(" %", "")) if parts[2] != "[N/A]" else 0,
                "memory_used": int(parts[3].replace(" MiB", "")) if "[N/A]" not in parts[3] else 0,
            }
        except Exception as e:
            log.warning(f"GPU {gpu_id} metrics failed: {e}")
            return {"temperature": 0, "power": 0, "utilization": 0, "memory_used": 0}
    
    def start_mining(self, gpu_id: int, wallet: str, power_limit: int = 100) -> float:
        """Start mining on GPU, return expected hashrate."""
        if gpu_id in self._mining_procs and self._mining_procs[gpu_id].poll() is None:
            log.info(f"GPU {gpu_id} already mining")
            return self._estimate_hashrate(gpu_id)
        
        # Build miner command (Keryx example)
        cmd = [
            "./keryx-miner",
            "-a", wallet,
            "-s", "127.0.0.1",
            "-p", "22111",
            "--cuda-device", str(gpu_id),
            "--cuda-workload", "256",
            "--cuda-no-blocking-sync",
            "--threads", "4",
        ]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self._mining_procs[gpu_id] = proc
            log.info(f"GPU {gpu_id} mining started (PID {proc.pid})")
            return self._estimate_hashrate(gpu_id)
        except Exception as e:
            log.error(f"GPU {gpu_id} mining start failed: {e}")
            return 0.0
    
    def stop_mining(self, gpu_id: int):
        """Stop mining on GPU."""
        proc = self._mining_procs.get(gpu_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
                log.info(f"GPU {gpu_id} mining stopped")
            except Exception as e:
                log.warning(f"GPU {gpu_id} mining stop failed: {e}")
                try:
                    proc.kill()
                except:
                    pass
        self._mining_procs.pop(gpu_id, None)
    
    def start_inference(self, gpu_id: int, model: str = "llama-3.2-1b"):
        """Start AI inference server on GPU."""
        if gpu_id in self._inference_procs and self._inference_procs[gpu_id].poll() is None:
            log.info(f"GPU {gpu_id} inference already running")
            return
        
        # Option 1: llama.cpp server
        cmd = [
            "./llama-server",
            "-m", f"models/{model}.gguf",
            "--port", str(8080 + gpu_id),
            "-ngl", "99",  # all layers to GPU
            "--ctx-size", "2048",
        ]
        
        # Option 2: vLLM (if installed)
        # cmd = ["python", "-m", "vllm.entrypoints.openai.api_server",
        #        "--model", model, "--port", str(8080 + gpu_id),
        #        "--tensor-parallel-size", "1"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self._inference_procs[gpu_id] = proc
            log.info(f"GPU {gpu_id} inference started on port {8080 + gpu_id}")
        except Exception as e:
            log.error(f"GPU {gpu_id} inference start failed: {e}")
    
    def stop_inference(self, gpu_id: int):
        """Stop inference server on GPU."""
        proc = self._inference_procs.get(gpu_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
                log.info(f"GPU {gpu_id} inference stopped")
            except Exception as e:
                log.warning(f"GPU {gpu_id} inference stop failed: {e}")
                try:
                    proc.kill()
                except:
                    pass
        self._inference_procs.pop(gpu_id, None)
    
    def is_mining(self, gpu_id: int) -> bool:
        """Check if mining is active on GPU."""
        proc = self._mining_procs.get(gpu_id)
        return proc is not None and proc.poll() is None
    
    def is_inferencing(self, gpu_id: int) -> bool:
        """Check if inference is active on GPU."""
        proc = self._inference_procs.get(gpu_id)
        return proc is not None and proc.poll() is None
    
    def _estimate_hashrate(self, gpu_id: int) -> float:
        """Estimate hashrate based on GPU model."""
        metrics = self.get_metrics(gpu_id)
        # Rough estimates by GPU tier
        name = self.discover()[gpu_id]["name"] if gpu_id < len(self.discover()) else ""
        if "5090" in name:
            return 2.65
        elif "4090" in name:
            return 1.95
        elif "3090" in name:
            return 1.2
        else:
            return 0.5  # fallback
