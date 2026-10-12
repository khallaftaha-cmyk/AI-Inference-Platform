"""
ecg_runner.py — Loads the ECG ONNX model and runs inference.

Key concepts in this file:
  - ONNX Runtime (ort): A fast engine for running .onnx model files.
    Instead of needing PyTorch + your full training setup, ONNX Runtime
    is a lightweight, production-optimized engine that just runs the model.

  - "Session": Think of it as "opening" the model file and getting it
    ready to receive inputs. You do this ONCE at startup, not on every request.
    Loading a model takes ~500ms; running inference takes ~5ms. So we load once.

  - Singleton pattern: We use a class with a single instance so the model
    is loaded only once, shared by all API requests.
"""

import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional

# Set up a logger for this module.
# Instead of print(), professional apps use loggers:
# → They can write to files, show timestamps, filter by severity, etc.
logger = logging.getLogger(__name__)


class ECGModelRunner:
    """
    Manages loading and running the ECG ONNX model.

    Usage:
        runner = ECGModelRunner(model_path="models_store/ecg_model_v1.onnx")
        result = runner.predict(signal_array)
    """

    def __init__(self, model_path: str, model_version: str = "v1"):
        """
        Initialize and load the model.

        Args:
            model_path: Full path to the .onnx file.
            model_version: Label like "v1" or "v2" used in logs and metrics.
        """
        self.model_path = model_path
        self.model_version = model_version
        self.session: Optional[ort.InferenceSession] = None  # loaded below

        self._load_model()

    def _load_model(self):
        """
        Internal method: Loads the ONNX model into memory.
        The underscore (_) prefix means "private — don't call this from outside."
        """
        path = Path(self.model_path)

        # Sanity check: Does the file even exist?
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Make sure you placed the .onnx file in models_store/"
            )

        logger.info(f"Loading ECG model {self.model_version} from: {self.model_path}")

        # SessionOptions lets us tune how ONNX Runtime uses your CPU.
        options = ort.SessionOptions()

        # These two settings control parallelism:
        # - intra_op: threads used WITHIN a single operation (e.g., matrix multiply)
        # - inter_op: threads used to run multiple operations in PARALLEL
        # Setting both to 1 is safe for a single-threaded API. Tune later.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        # Enable graph optimization: ONNX Runtime rewrites the model graph
        # to fuse operations and remove redundancies → faster inference.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create the inference session (this is the "loading" step).
        # ["CPUExecutionProvider"] = use CPU. Later you could add CUDA for GPU.
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        # Log what inputs the model expects — useful for debugging.
        for inp in self.session.get_inputs():
            logger.info(
                f"  Input name: '{inp.name}' | Shape: {inp.shape} | Type: {inp.type}"
            )

        logger.info(f"[OK] ECG model {self.model_version} loaded successfully.")

    def predict(self, signal: np.ndarray) -> dict:
        """
        Run inference on an ECG signal.

        Args:
            signal: A numpy array representing the ECG signal.
                    Expected shape: (1, num_leads, signal_length)
                    Example: (1, 12, 1000) for a 12-lead, 1000-sample ECG.

        Returns:
            A dict with:
              - "probabilities": list of class probabilities (one per class)
              - "predicted_class": the class index with highest probability
              - "confidence": the highest probability (0.0 to 1.0)
              - "model_version": which model made this prediction
        """
        if self.session is None:
            raise RuntimeError("Model session is not initialized. Call _load_model() first.")

        # ONNX Runtime needs inputs as a dict: {input_name: numpy_array}
        # We get the input name from the session itself (no hardcoding).
        input_name = self.session.get_inputs()[0].name

        # Ensure the array is float32 — ONNX models typically expect float32.
        # float64 (Python's default) would cause a type mismatch error.
        signal = signal.astype(np.float32)

        # Run inference!
        # outputs is a list of numpy arrays (one per output node in the model).
        outputs = self.session.run(None, {input_name: signal})

        # outputs[0] is the raw logits or probabilities from the model.
        # Shape is typically (1, num_classes) — we take [0] to remove the batch dim.
        raw_output = outputs[0][0]  # shape: (num_classes,)

        # If raw_output isn't already probabilities (softmax), we apply softmax.
        # softmax converts raw scores into probabilities that sum to 1.
        # We check if they already sum to ~1 (already probabilities).
        if not (0.99 < raw_output.sum() < 1.01):
            # Apply softmax: e^x_i / sum(e^x_j)
            exp_output = np.exp(raw_output - raw_output.max())  # subtract max for stability
            probabilities = exp_output / exp_output.sum()
        else:
            probabilities = raw_output

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return {
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "model_version": self.model_version,
        }

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready."""
        return self.session is not None
