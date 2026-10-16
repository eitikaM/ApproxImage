"""
adaptive_inference.py
=====================
Phase 6: Application-Aware Adaptive Approximate Computing.
Standalone ML Inference Pipeline for Runtime Multiplier Mode Selection.

Inputs:
  - image: File path (str) or 2D/3D numpy array.
  - effect: 'blur', 'edge', 'sharpen', or 'emboss'.

Steps:
  1. Loads image and converts to 2D uint8 grayscale.
  2. Extracts the exact 5 physical and statistical features used during training:
     - mean_intensity
     - contrast (std dev)
     - edge_density (fraction with gradient > 25.0)
     - average_gradient (mean L2 gradient magnitude)
     - texture (std dev of discrete 3x3 Laplacian)
  3. Encodes the task effect (one-hot aligned with trained model schema).
  4. Computes model posterior probability of Mode 3 being unsafe: P(unsafe = 1).
  5. Applies stored conservative safety threshold (default = 0.12).
  6. Decision Logic:
     - If P(unsafe) >= threshold: Select Mode 2 (Med approx, safe fallback).
     - Else: Select Mode 3 (High approx, maximum efficiency).

Returns:
  - selected_mode: int (2 or 3)
  - probability_unsafe: float
  - extracted_features: dict
  - reason: str
"""

import os
import sys
import argparse
import joblib
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "ml", "trained_model.joblib")

VALID_EFFECTS = ["blur", "edge", "sharpen", "emboss"]


class AdaptiveInferenceEngine:
    """Production-ready inference engine for adaptive approximate multiplier mode selection."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model bundle not found at: {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.threshold = float(bundle.get("decision_threshold", 0.12))
        self.metadata = bundle.get("metadata", {})

    @staticmethod
    def load_and_preprocess_image(image_input) -> np.ndarray:
        """
        Loads an image from file path or validates an input numpy array.
        Returns a 2D uint8 grayscale array.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image at: {image_input}")
            return img

        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                # Color BGR or RGB to grayscale
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif len(image_input.shape) == 2:
                gray = image_input
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")

            if gray.dtype != np.uint8:
                gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
            return gray

        raise TypeError(f"image_input must be file path (str) or np.ndarray, got {type(image_input)}")

    @staticmethod
    def extract_features(gray_img: np.ndarray) -> dict:
        """
        Extracts exactly the 5 features used during model training:
        - mean_intensity: Mean pixel value [0, 255]
        - contrast: Standard deviation of pixel values
        - edge_density: Fraction of pixels with gradient magnitude > 25.0
        - average_gradient: Mean gradient magnitude (L2 norm of finite differences)
        - texture: Standard deviation of discrete 3x3 Laplacian
        """
        assert gray_img.dtype == np.uint8 and len(gray_img.shape) == 2, "Image must be 2D uint8."
        img_f = gray_img.astype(np.float64)

        # 1. Mean intensity
        mean_intensity = float(np.mean(img_f))

        # 2. Contrast
        contrast = float(np.std(img_f))

        # 3. Spatial Gradient & Edge Density
        gy, gx = np.gradient(img_f)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        average_gradient = float(np.mean(grad_mag))
        edge_density = float(np.mean(grad_mag > 25.0))

        # 4. Texture (Laplacian variation)
        padded = np.pad(img_f, pad_width=1, mode="reflect")
        H, W = gray_img.shape
        lap = (
            padded[0:H, 1:W + 1] + padded[2:H + 2, 1:W + 1] +
            padded[1:H + 1, 0:W] + padded[1:H + 1, 2:W + 2] -
            4.0 * padded[1:H + 1, 1:W + 1]
        )
        texture = float(np.std(lap))

        return {
            "mean_intensity": round(mean_intensity, 4),
            "contrast": round(contrast, 4),
            "edge_density": round(edge_density, 4),
            "average_gradient": round(average_gradient, 4),
            "texture": round(texture, 4)
        }

    def predict(self, image_input, effect: str) -> dict:
        """
        Selects optimal multiplier mode (Mode 2 or Mode 3) for the given image and effect.
        """
        effect_clean = effect.lower().strip()
        if effect_clean not in VALID_EFFECTS:
            raise ValueError(f"Invalid effect '{effect}'. Must be one of: {VALID_EFFECTS}")

        # 1. Load and prepare image
        gray_img = self.load_and_preprocess_image(image_input)

        # 2. Extract features
        feats = self.extract_features(gray_img)

        # 3. Build feature row matching trained model schema
        feature_dict = feats.copy()
        for eff in VALID_EFFECTS:
            feature_dict[f"effect_{eff}"] = 1.0 if eff == effect_clean else 0.0

        feature_vector = np.array([[feature_dict[c] for c in self.feature_cols]], dtype=np.float64)

        # 4. Predict posterior probability P(unsafe = 1)
        prob_unsafe = float(self.model.predict_proba(feature_vector)[0, 1])

        # 5. Apply conservative decision threshold
        if prob_unsafe >= self.threshold:
            selected_mode = 2
            reason = (
                f"Mode 3 is predicted UNSAFE with probability {prob_unsafe * 100:.1f}% "
                f"(>= safety threshold {self.threshold * 100:.1f}%). "
                f"Falling back to Mode 2 to strictly preserve SSIM >= 0.99 quality constraint."
            )
        else:
            selected_mode = 3
            reason = (
                f"Mode 3 is predicted SAFE with probability {(1.0 - prob_unsafe) * 100:.1f}% "
                f"(unsafe probability {prob_unsafe * 100:.1f}% < threshold {self.threshold * 100:.1f}%). "
                f"Selecting Mode 3 for maximum hardware savings (28.42% logic cell reduction)."
            )

        return {
            "selected_mode": selected_mode,
            "probability_unsafe": round(prob_unsafe, 4),
            "extracted_features": feats,
            "decision_threshold": self.threshold,
            "effect": effect_clean,
            "reason": reason
        }


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multiplier Mode Selector CLI")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--effect", required=True, choices=VALID_EFFECTS, help="Target image filter effect")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model bundle (.joblib)")
    args = parser.parse_args()

    engine = AdaptiveInferenceEngine(model_path=args.model)
    result = engine.predict(args.image, args.effect)

    print("=" * 70)
    print("      ADAPTIVE APPROXIMATE MULTIPLIER MODE SELECTOR RESULT")
    print("=" * 70)
    print(f"Image Path        : {args.image}")
    print(f"Filter Effect     : {result['effect'].capitalize()}")
    print(f"Selected Mode     : Mode {result['selected_mode']} ({'Medium Approximation' if result['selected_mode'] == 2 else 'High Approximation'})")
    print(f"Hardware Savings  : {'13.39%' if result['selected_mode'] == 2 else '28.42%'} logic cells")
    print(f"Unsafe Probability: {result['probability_unsafe'] * 100:.2f}% (Threshold: {result['decision_threshold'] * 100:.2f}%)")
    print("\nExtracted Features:")
    for k, v in result["extracted_features"].items():
        print(f"  - {k:<18}: {v}")
    print(f"\nDecision Rationale:\n  {result['reason']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
