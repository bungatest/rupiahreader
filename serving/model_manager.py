"""
Model Manager for Rupiah Banknote Serving.
Handles thread-safe inference, model loading, registry queries,
and consensus aggregation across CNN, MobileNetV2, and FNN models.
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from .preprocessing import preprocess_for_model, aggregate_frame_probabilities
from .config import (
    SERVING_MODELS_DIR,
    CLASSES,
    CLASS_DISPLAY_NAMES,
    SPOKEN_TEXT,
    DECISION_THRESHOLDS,
)

# Global lock for TF/Keras model thread-safety
TF_INFERENCE_LOCK = threading.Lock()


class ModelManager:
    def __init__(self, models_dir: Path = SERVING_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, tf.keras.Model] = {}
        self.registry: Dict[str, Any] = {}
        self.load_registry()
        self.load_available_models()

    def load_registry(self) -> None:
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                print(f"[ModelManager] Warning: failed to parse registry.json: {e}")
                self.registry = {}
        else:
            self.registry = {}

    def load_available_models(self) -> None:
        """Scan and load trained models with lock."""
        with TF_INFERENCE_LOCK:
            # 1. Custom CNN
            cnn_path = self.models_dir / "custom_cnn_selected_final.keras"
            if cnn_path.exists():
                try:
                    print(f"[ModelManager] Loading Custom CNN from {cnn_path}...")
                    self.models["cnn"] = tf.keras.models.load_model(str(cnn_path))
                    print(f"[ModelManager] Custom CNN loaded successfully.")
                except Exception as e:
                    print(f"[ModelManager] Error loading Custom CNN: {e}")

            # 2. MobileNetV2
            mob_path = self.models_dir / "mobilenetv2_finetuned_final.keras"
            if mob_path.exists():
                try:
                    print(f"[ModelManager] Loading MobileNetV2 from {mob_path}...")
                    self.models["mobilenet"] = tf.keras.models.load_model(str(mob_path))
                    print(f"[ModelManager] MobileNetV2 loaded successfully.")
                except Exception as e:
                    print(f"[ModelManager] Error loading MobileNetV2: {e}")

            # 3. FNN
            fnn_path = self.models_dir / "fnn_selected_final.keras"
            if fnn_path.exists():
                try:
                    print(f"[ModelManager] Loading FNN from {fnn_path}...")
                    self.models["fnn"] = tf.keras.models.load_model(str(fnn_path))
                    print(f"[ModelManager] FNN loaded successfully.")
                except Exception as e:
                    print(f"[ModelManager] Error loading FNN: {e}")

    def get_models_status(self) -> List[Dict[str, Any]]:
        """Return list of models with their active/inactive status and registry metadata."""
        model_defs = [
            ("cnn", "Custom CNN (8-Layer ConvNet)", "Rupiah_CustomCNN.ipynb"),
            ("mobilenet", "MobileNetV2 (Fine-Tuned Transfer Learning)", "mobilenetv2-training-kaggle.ipynb"),
            ("fnn", "FNN / MLP Flatten", "Rupiah_FNN.ipynb"),
        ]
        results = []
        for mid, name, nb in model_defs:
            is_loaded = mid in self.models
            meta = self.registry.get(mid, {})
            results.append({
                "id": mid,
                "name": name,
                "notebook": nb,
                "loaded": is_loaded,
                "status": "tersedia" if is_loaded else "tidak_tersedia",
                "threshold": DECISION_THRESHOLDS.get(mid, 0.70),
                "metadata": meta
            })
        return results

    def predict_single(self, model_id: str, image: Image.Image) -> Dict[str, Any]:
        """Run inference with a single model."""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' is not loaded or not available.")

        model = self.models[model_id]
        x = preprocess_for_model(image, model_id)

        t0 = time.time()
        with TF_INFERENCE_LOCK:
            probs = model.predict(x, verbose=0)[0]
        latency_ms = round((time.time() - t0) * 1000, 1)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        threshold = DECISION_THRESHOLDS.get(model_id, 0.70)

        # Sorted top predictions
        sorted_indices = np.argsort(probs)[::-1]
        top3 = [
            {
                "class": CLASSES[i],
                "display": CLASS_DISPLAY_NAMES[CLASSES[i]],
                "probability": round(float(probs[i]), 4),
            }
            for i in sorted_indices[:3]
        ]

        # Margin top1 - top2
        top1_p = float(probs[sorted_indices[0]])
        top2_p = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = round(top1_p - top2_p, 4)

        is_confident = (confidence >= threshold) and (margin >= 0.15)
        status = "pasti" if is_confident else "tidak_yakin"

        spoken = SPOKEN_TEXT.get(pred_class, pred_class) if is_confident else "Uang tidak yakin, silakan coba dekatkan atau rapikan uang kertas."

        return {
            "model_id": model_id,
            "predicted_class": pred_class if is_confident else "tidak_yakin",
            "predicted_display": CLASS_DISPLAY_NAMES[pred_class] if is_confident else "Tidak Yakin",
            "raw_class": pred_class,
            "raw_display": CLASS_DISPLAY_NAMES[pred_class],
            "confidence": round(confidence, 4),
            "margin": margin,
            "threshold": threshold,
            "status": status,
            "audio_text": spoken,
            "latency_ms": latency_ms,
            "top3": top3,
            "all_probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        }

    def predict_multi_frame(self, model_id: str, frames: List[Image.Image]) -> Dict[str, Any]:
        """Aggregate predictions over multiple frames."""
        if not frames:
            raise ValueError("No frames provided for prediction.")
        if len(frames) == 1:
            return self.predict_single(model_id, frames[0])

        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' is not loaded or not available.")

        model = self.models[model_id]
        batch = np.concatenate([preprocess_for_model(img, model_id) for img in frames], axis=0)

        t0 = time.time()
        with TF_INFERENCE_LOCK:
            batch_probs = model.predict(batch, verbose=0)
        latency_ms = round((time.time() - t0) * 1000, 1)

        avg_probs = np.mean(batch_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(avg_probs[pred_idx])
        threshold = DECISION_THRESHOLDS.get(model_id, 0.70)

        sorted_indices = np.argsort(avg_probs)[::-1]
        top3 = [
            {
                "class": CLASSES[i],
                "display": CLASS_DISPLAY_NAMES[CLASSES[i]],
                "probability": round(float(avg_probs[i]), 4),
            }
            for i in sorted_indices[:3]
        ]

        top1_p = float(avg_probs[sorted_indices[0]])
        top2_p = float(avg_probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = round(top1_p - top2_p, 4)

        is_confident = (confidence >= threshold) and (margin >= 0.15)
        status = "pasti" if is_confident else "tidak_yakin"
        spoken = SPOKEN_TEXT.get(pred_class, pred_class) if is_confident else "Uang tidak yakin, coba dekatkan kembali."

        return {
            "model_id": model_id,
            "predicted_class": pred_class if is_confident else "tidak_yakin",
            "predicted_display": CLASS_DISPLAY_NAMES[pred_class] if is_confident else "Tidak Yakin",
            "raw_class": pred_class,
            "raw_display": CLASS_DISPLAY_NAMES[pred_class],
            "confidence": round(confidence, 4),
            "margin": margin,
            "threshold": threshold,
            "status": status,
            "audio_text": spoken,
            "frames_processed": len(frames),
            "latency_ms": latency_ms,
            "top3": top3,
            "all_probabilities": {CLASSES[i]: round(float(avg_probs[i]), 4) for i in range(len(CLASSES))},
        }

    def predict_consensus(self, image_or_frames: Any) -> Dict[str, Any]:
        """
        Run inference across ALL available models and aggregate by consensus.
        - Requires agreement from >= 60% of available models
        - Average probability across agreeing models >= consensus threshold
        - Margin (top1 - top2) >= 0.20
        """
        if not self.models:
            raise RuntimeError("No models are currently loaded and available for serving.")

        frames = image_or_frames if isinstance(image_or_frames, list) else [image_or_frames]
        per_model_results = {}
        votes = {}

        t_start = time.time()
        for mid in self.models.keys():
            res = self.predict_multi_frame(mid, frames)
            per_model_results[mid] = res
            raw_c = res["raw_class"]
            votes[raw_c] = votes.get(raw_c, []) + [res]
        total_latency_ms = round((time.time() - t_start) * 1000, 1)

        n_models = len(self.models)
        # Find winning class with most votes
        sorted_votes = sorted(votes.items(), key=lambda kv: len(kv[1]), reverse=True)
        top_class, top_res_list = sorted_votes[0]
        agree_count = len(top_res_list)
        agreement_ratio = agree_count / n_models

        # Consensus conditions
        avg_confidence = float(np.mean([r["confidence"] for r in top_res_list]))
        consensus_threshold = DECISION_THRESHOLDS.get("consensus", 0.75)

        # Average margin
        avg_margin = float(np.mean([r["margin"] for r in top_res_list]))

        is_agreed = (agreement_ratio >= 0.60) and (avg_confidence >= consensus_threshold) and (avg_margin >= 0.20)

        if is_agreed:
            final_class = top_class
            status = "pasti"
            spoken = SPOKEN_TEXT.get(final_class, final_class)
            reason = f"Konsensus dicapai: {agree_count}/{n_models} model sepakat ({agreement_ratio:.0%}) dengan rata-rata keyakinan {avg_confidence:.1%}"
        else:
            final_class = "tidak_yakin"
            status = "tidak_yakin"
            spoken = "Hasil belum meyakinkan, coba sesuaikan posisi atau pencahayaan uang kertas."
            if agreement_ratio < 0.60:
                reason = f"Model tidak sepakat (suara terbanyak hanya {agree_count}/{n_models})."
            elif avg_confidence < consensus_threshold:
                reason = f"Keyakinan rata-rata ({avg_confidence:.1%}) di bawah ambang batas konsensus ({consensus_threshold:.1%})."
            else:
                reason = f"Margin probabilitas tipis ({avg_margin:.2f} < 0.20)."

        return {
            "mode": "consensus",
            "predicted_class": final_class,
            "predicted_display": CLASS_DISPLAY_NAMES.get(final_class, "Tidak Yakin"),
            "agreement_ratio": round(agreement_ratio, 2),
            "agree_count": agree_count,
            "total_models": n_models,
            "average_confidence": round(avg_confidence, 4),
            "average_margin": round(avg_margin, 4),
            "status": status,
            "audio_text": spoken,
            "consensus_reason": reason,
            "total_latency_ms": total_latency_ms,
            "per_model": per_model_results,
        }

