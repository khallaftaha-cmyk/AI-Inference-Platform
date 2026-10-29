"""
registry.py — Dynamic Model Registry & Lifecycle Manager.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from app.models.ecg_runner import ECGModelRunner

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages the collection of loaded and available models.
    """

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self._loaded_models: Dict[str, ECGModelRunner] = {}
        self._active_version: Optional[str] = None
        self._lock = threading.Lock()

    def discover_models(self) -> List[dict]:
        """
        Scans models_dir for all .onnx files and returns metadata about each.
        """
        if not self.models_dir.exists():
            return []

        models_info = []
        for file in sorted(self.models_dir.glob("*.onnx")):
            version = file.stem  # e.g., "ecg_model_v1"
            stat = file.stat()
            is_loaded = version in self._loaded_models
            is_active = (version == self._active_version or (self._active_version and self._active_version in version))

            models_info.append({
                "version": version,
                "filename": file.name,
                "file_size_bytes": stat.st_size,
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_loaded": is_loaded,
                "is_active": is_active,
            })

        return models_info

    def load_model(self, version: str) -> ECGModelRunner:
        """
        Loads an ONNX model file into RAM if not already loaded.
        Thread-safe to prevent race conditions during concurrent loads.
        """
        with self._lock:
            if version in self._loaded_models:
                logger.info(f"Model {version} already loaded in memory.")
                return self._loaded_models[version]

            # Find matching file (e.g. version "v1" -> "ecg_model_v1.onnx" or "v1.onnx")
            candidate_files = [
                self.models_dir / f"{version}.onnx",
                self.models_dir / f"ecg_model_{version}.onnx",
            ]
            
            target_file = None
            for f in candidate_files:
                if f.exists():
                    target_file = f
                    break

            if target_file is None:
                direct_match = self.models_dir / f"{version}"
                if direct_match.suffix == ".onnx" and direct_match.exists():
                    target_file = direct_match

            if target_file is None:
                raise FileNotFoundError(
                    f"Model '{version}' not found in {self.models_dir}. "
                    f"Available: {[f.name for f in self.models_dir.glob('*.onnx')]}"
                )

            logger.info(f"Pre-warming & loading model '{version}' from {target_file}...")
            runner = ECGModelRunner(model_path=str(target_file), model_version=version)
            self._loaded_models[version] = runner

            # Register stem as alias if different (e.g., "ecg_model_v1" <-> "v1")
            file_stem = target_file.stem
            if file_stem != version:
                self._loaded_models[file_stem] = runner

            # If this is the first model loaded, make it active by default
            if self._active_version is None:
                self._active_version = version
                logger.info(f"Set '{version}' as active production model.")

            return runner

    def activate_model(self, version: str) -> None:
        """
        Atomically switches the active production model to the given version.
        """
        with self._lock:
            if version not in self._loaded_models:
                self.load_model(version)

            old_version = self._active_version
            self._active_version = version
            logger.info(f"[SWITCH] Active model switched: {old_version} -> {version}")

    def unload_model(self, version: str) -> bool:
        """
        Removes a model runner from RAM to release resources.
        Cannot unload the currently active production model.
        """
        with self._lock:
            if version == self._active_version or (self._active_version and self._active_version in version):
                raise ValueError(
                    f"Cannot unload active model '{version}'. "
                    f"Activate a different model before unloading this one."
                )

            unloaded = False
            if version in self._loaded_models:
                del self._loaded_models[version]
                unloaded = True

            # Also clean up potential aliases
            aliases = [k for k, v in list(self._loaded_models.items()) if k == f"ecg_model_{version}" or k.replace("ecg_model_", "") == version]
            for a in aliases:
                del self._loaded_models[a]

            if unloaded:
                logger.info(f"Unloaded model '{version}' from memory.")
            return unloaded

    def get_active_model(self) -> Optional[ECGModelRunner]:
        """Returns the currently active production model runner."""
        if self._active_version and self._active_version in self._loaded_models:
            return self._loaded_models[self._active_version]
        return None

    def get_model(self, version: str) -> Optional[ECGModelRunner]:
        """Returns a specific loaded model runner by version name."""
        return self._loaded_models.get(version)

    @property
    def active_version(self) -> Optional[str]:
        return self._active_version

    @property
    def loaded_versions(self) -> List[str]:
        # Return unique loaded version keys
        return sorted(list(set(self._loaded_models.keys())))
