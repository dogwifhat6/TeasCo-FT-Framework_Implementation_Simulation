"""Environment probes the UI uses to explain what will and will not run here."""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional


def total_memory_gb() -> Optional[float]:
    """Physical memory in GB, or None if it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5)
        return int(out.stdout.strip()) / 1024 ** 3
    except Exception:
        return None


def ollama_models(host: str = "http://localhost:11434") -> Optional[List[str]]:
    """Locally pulled Ollama model tags, or None if the server is unreachable."""
    try:
        import requests

        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=3)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:
        return None


def load_env(path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding the environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass
