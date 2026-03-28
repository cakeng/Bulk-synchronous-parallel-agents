#!/usr/bin/env python3
"""Launch script for vLLM server using configuration from config/vllm_config.yaml"""

import sys
import signal
import time
import os
import sys
import json
import time
import subprocess
import signal
import requests
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer

class VLLMServer:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, config: dict):
        """Initialize vLLM server manager.
        
        Args:
            config_path: Optional path to vLLM config file
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{self.config['server']['host']}:{self.config['server']['port']}"
        
    def _build_vllm_command(self) -> list:
        """Build the command to start vLLM server."""
        # Use the current interpreter to avoid accidentally launching vLLM under
        # a different environment than the one importing this module.
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        
        # Model configuration
        model_name = self.config['model']['name']
        cmd.extend(["--model", model_name])
        
        if self.config['model'].get('trust_remote_code', False):
            cmd.append("--trust-remote-code")
        
        # Server configuration
        cmd.extend(["--host", self.config['server']['host']])
        cmd.extend(["--port", str(self.config['server']['port'])])
        
        # GPU configuration
        if self.config.get('tensor_parallel_size', 1) > 1:
            cmd.extend(["--tensor-parallel-size", str(self.config['tensor_parallel_size'])])
        
        if 'gpu_memory_utilization' in self.config['gpu']:
            cmd.extend(["--gpu-memory-utilization", str(self.config['gpu']['gpu_memory_utilization'])])
        
        # Model context length
        if self.config.get('max_model_len'):
            cmd.extend(["--max-model-len", str(self.config['max_model_len'])])
        
        # KV cache dtype
        if self.config.get('kv_cache_dtype'):
            cmd.extend(["--kv-cache-dtype", str(self.config['kv_cache_dtype'])])
        
        # Additional arguments
        if self.config.get('additional_args'):
            cmd.extend(self.config['additional_args'])
        
        return cmd

    def start(self) -> bool:
        """Start the vLLM server.
        """
        if self.is_running():
            print(f"vLLM server is already running at {self.base_url}")
            return True
        
        cmd = self._build_vllm_command()
        
        # Set up environment variables
        env = os.environ.copy()
        
        # Set CUDA_VISIBLE_DEVICES if GPU IDs are specified
        gpu_ids = self.config.get('gpu', {}).get('gpu_ids')
        if gpu_ids is not None:
            if isinstance(gpu_ids, list):
                cuda_visible_devices = ','.join(str(gpu_id) for gpu_id in gpu_ids)
            else:
                cuda_visible_devices = str(gpu_ids)
            env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
            print(f"Setting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        
        print(f"Starting vLLM server with command: {' '.join(cmd)}")
        
        # Start the server process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,  # Print vLLM output to stdout
                stderr=sys.stderr,  # Print vLLM errors to stderr
                text=True,
                env=env
            )
        except Exception as e:
            print(f"Failed to start vLLM server: {e}")
            return False
        
        return True

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.process:
            print("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing vLLM server...")
                self.process.kill()
                self.process.wait()
            self.process = None
            print("vLLM server stopped")
    
    def is_running(self) -> bool:
        """Check if the vLLM server is running.
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

def main():
    """Main entry point for running vLLM server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start vLLM server")
    parser.add_argument(
        "--config",
        type=str,
        default="vllm_config.yaml",
        help="Path to vLLM configuration file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    server = VLLMServer(config)

    try:
        if server.start():
            print("vLLM server started. Press Ctrl+C to stop.")
            # Keep the server running
            try:
                while True:
                    time.sleep(5)
                    if server.process and server.process.poll() is not None:
                        break
            except KeyboardInterrupt:
                pass
        else:
            print("Failed to start vLLM server")
            sys.exit(1)
    finally:
        server.stop()

if __name__ == "__main__":
    main()
