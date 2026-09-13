"""Hash-pinned Ultralytics YOLO11 segmentation backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .provider import InstancePrediction, normalize_names, sha256_file


class UltralyticsBackend:
    """Load one trusted, hash-pinned segmentation checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        expected_sha256: str,
        *,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        actual = sha256_file(self.checkpoint)
        if actual != expected:
            raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        # Import only after the explicit user-provided path and SHA have been
        # checked. Ultralytics .pt files are pickle-bearing Python archives.
        from ultralytics import YOLO  # type: ignore

        self.checkpoint_sha256 = actual
        self.device = str(device)
        self.image_size = int(image_size)
        self.model = YOLO(str(self.checkpoint), task="segment")
        if str(getattr(self.model, "task", "")) != "segment":
            raise RuntimeError("checkpoint is not an Ultralytics segmentation model")
        self.names = normalize_names(self.model.names)
        if not self.names:
            raise RuntimeError("checkpoint has no class-name table")

    def predict(
        self, rgb: np.ndarray, minimum_confidence: float
    ) -> tuple[list[InstancePrediction], dict[str, Any]]:
        """Infer on canonical RGB and return masks on the original image grid."""
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("input must be HxWx3 RGB uint8")
        started = time.perf_counter()
        # Ultralytics numpy input follows OpenCV's BGR convention.
        bgr = np.ascontiguousarray(image[..., ::-1])
        results = self.model.predict(
            source=bgr,
            conf=float(minimum_confidence),
            imgsz=self.image_size,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(results) != 1:
            raise RuntimeError(f"expected one inference result, got {len(results)}")
        result = results[0]
        predictions: list[InstancePrediction] = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            masks = result.masks.data.detach().cpu().numpy()
            if not (len(classes) == len(confidences) == len(masks)):
                raise RuntimeError("model boxes and masks have different counts")
            for class_id_value, confidence, mask in zip(classes, confidences, masks):
                class_id = int(class_id_value)
                predictions.append(
                    InstancePrediction(
                        class_id=class_id,
                        label=self.names.get(class_id, f"class_{class_id}"),
                        confidence=float(confidence),
                        mask_probability=np.asarray(mask, dtype=np.float32),
                    )
                )
        return predictions, {
            "inference_ms": elapsed_ms,
            "device": self.device,
            "image_size": self.image_size,
            "model_names": self.names,
            "raw_instance_count": len(predictions),
        }
