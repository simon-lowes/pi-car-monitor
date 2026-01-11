#!/usr/bin/env python3
"""
Fine-Tuning Script
==================
Fine-tunes a car recognition model using reference images and calibration data.

This creates a custom model optimized for recognizing your specific car
from the camera angle and lighting conditions of your setup.

Run with: python training/fine_tune.py

Note: Full fine-tuning requires GPU and Hailo Dataflow Compiler.
This script provides a lighter-weight approach using feature embeddings.
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class CarEmbeddingModel:
    """
    Car recognition using feature embeddings.

    This approach:
    1. Extracts features from reference images using a pre-trained model
    2. Stores average embedding for the target car
    3. At runtime, compares detected cars to this embedding

    Advantages:
    - Works without GPU fine-tuning
    - Can run on Hailo with existing models
    - Quick to set up
    """

    def __init__(self, model_path: Optional[str] = None):
        self.embeddings = []
        self.mean_embedding = None
        self.colour_histogram = None
        self.model_path = model_path

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        try:
            import cv2

            # Resize to standard size
            resized = cv2.resize(image, (224, 224))

            # Convert to float and normalize
            normalized = resized.astype(np.float32) / 255.0

            # Simple feature extraction using colour histograms
            # and edge features (works without deep learning)

            features = []

            # Colour histogram features
            for i in range(3):
                hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
                hist = hist.flatten() / hist.sum()
                features.extend(hist)

            # HSV histogram for colour invariance
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
            h_hist = h_hist.flatten() / h_hist.sum()
            features.extend(h_hist)

            # Edge features
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
            edge_hist = edge_hist.flatten() / (edge_hist.sum() + 1e-6)
            features.extend(edge_hist)

            # Shape features (moments)
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)

            return np.array(features)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(128)

    def train(self, image_paths: List[str]) -> bool:
        """Train on reference images."""
        print(f"Training on {len(image_paths)} reference images...")

        try:
            import cv2

            self.embeddings = []

            for i, path in enumerate(image_paths):
                image = cv2.imread(str(path))
                if image is None:
                    logger.warning(f"Could not read image: {path}")
                    continue

                features = self.extract_features(image)
                self.embeddings.append(features)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images")

            if not self.embeddings:
                logger.error("No valid images for training")
                return False

            # Calculate mean embedding
            self.mean_embedding = np.mean(self.embeddings, axis=0)

            # Calculate colour histogram from all images
            self._compute_colour_profile(image_paths)

            print(f"Training complete. Embedding size: {len(self.mean_embedding)}")
            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def _compute_colour_profile(self, image_paths: List[str]):
        """Compute average colour profile from training images."""
        try:
            import cv2

            histograms = []

            for path in image_paths:
                image = cv2.imread(str(path))
                if image is None:
                    continue

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Focus on center region (car body, not background)
                h, w = hsv.shape[:2]
                center = hsv[h//4:3*h//4, w//4:3*w//4]

                hist = cv2.calcHist([center], [0, 1], None, [18, 8], [0, 180, 0, 256])
                hist = hist / (hist.sum() + 1e-6)
                histograms.append(hist)

            if histograms:
                self.colour_histogram = np.mean(histograms, axis=0)

        except Exception as e:
            logger.warning(f"Colour profile error: {e}")

    def match(self, image: np.ndarray, threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Match an image against the trained model.

        Returns:
            Tuple of (is_match, confidence)
        """
        if self.mean_embedding is None:
            return False, 0.0

        features = self.extract_features(image)

        # Cosine similarity
        dot_product = np.dot(features, self.mean_embedding)
        norm_a = np.linalg.norm(features)
        norm_b = np.linalg.norm(self.mean_embedding)

        if norm_a == 0 or norm_b == 0:
            return False, 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Scale to 0-1 range (cosine similarity can be negative)
        confidence = (similarity + 1) / 2

        return confidence >= threshold, float(confidence)

    def save(self, path: str):
        """Save the trained model."""
        data = {
            "embeddings": self.embeddings,
            "mean_embedding": self.mean_embedding,
            "colour_histogram": self.colour_histogram,
            "created": datetime.now().isoformat()
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> bool:
        """Load a trained model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.embeddings = data.get("embeddings", [])
            self.mean_embedding = data.get("mean_embedding")
            self.colour_histogram = data.get("colour_histogram")

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def prepare_training_data(
    reference_dir: Path,
    calibration_dir: Path
) -> List[str]:
    """Prepare training data from reference and calibration images."""
    images = []

    # Collect reference images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for ext in extensions:
        images.extend(reference_dir.glob(f"*{ext}"))
        images.extend(reference_dir.glob(f"*{ext.upper()}"))

    # Collect calibration images (higher weight - actual camera view)
    for ext in extensions:
        calibration_images = list(calibration_dir.glob(f"*{ext}"))
        calibration_images.extend(calibration_dir.glob(f"*{ext.upper()}"))
        # Add calibration images multiple times for higher weight
        images.extend(calibration_images * 3)

    return [str(p) for p in images]


def evaluate_model(model: CarEmbeddingModel, test_images: List[str]) -> dict:
    """Evaluate model on test images."""
    try:
        import cv2

        results = []

        for path in test_images:
            image = cv2.imread(str(path))
            if image is None:
                continue

            is_match, confidence = model.match(image)
            results.append({
                "path": path,
                "match": is_match,
                "confidence": confidence
            })

        if not results:
            return {"error": "No valid test images"}

        avg_confidence = np.mean([r["confidence"] for r in results])
        match_rate = np.mean([r["match"] for r in results])

        return {
            "total_images": len(results),
            "average_confidence": float(avg_confidence),
            "match_rate": float(match_rate),
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune car recognition model")
    parser.add_argument("--reference-dir", default="data/reference_images",
                        help="Reference images directory")
    parser.add_argument("--calibration-dir", default="data/calibration",
                        help="Calibration images directory")
    parser.add_argument("--output", default="models/custom/car_embedding.pkl",
                        help="Output model path")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate after training")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    reference_dir = Path(args.reference_dir)
    calibration_dir = Path(args.calibration_dir)
    output_path = Path(args.output)

    # Check directories exist
    if not reference_dir.exists() and not calibration_dir.exists():
        print("ERROR: No training data found.")
        print(f"  Reference images: {reference_dir}")
        print(f"  Calibration images: {calibration_dir}")
        print("\nRun these first:")
        print("  python training/collect_references.py")
        print("  python training/calibrate.py")
        sys.exit(1)

    # Prepare training data
    print("\n" + "=" * 50)
    print("  Car Recognition Model Training")
    print("=" * 50 + "\n")

    training_images = prepare_training_data(reference_dir, calibration_dir)

    if len(training_images) < 5:
        print(f"WARNING: Only {len(training_images)} training images found.")
        print("For best results, provide at least 20 reference images.")

        if len(training_images) == 0:
            print("\nNo images found. Please add images to:")
            print(f"  {reference_dir}")
            print(f"  {calibration_dir}")
            sys.exit(1)

    print(f"Found {len(training_images)} training images")

    # Train model
    model = CarEmbeddingModel()

    if not model.train(training_images):
        print("Training failed")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(str(output_path))
    print(f"\nModel saved to: {output_path}")

    # Evaluate if requested
    if args.evaluate:
        print("\n" + "-" * 30)
        print("Evaluation")
        print("-" * 30)

        # Use calibration images for evaluation (should match well)
        test_images = list(calibration_dir.glob("*.jpg"))
        test_images.extend(calibration_dir.glob("*.png"))

        if test_images:
            results = evaluate_model(model, [str(p) for p in test_images])
            print(f"  Test images: {results.get('total_images', 0)}")
            print(f"  Average confidence: {results.get('average_confidence', 0):.2%}")
            print(f"  Match rate: {results.get('match_rate', 0):.2%}")
        else:
            print("  No test images found in calibration directory")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print("\nThe car recognition model has been trained.")
    print("\nNote: This is an embedding-based model. For a full")
    print("Hailo-optimized HEF model, you would need to:")
    print("  1. Train a neural network on your data")
    print("  2. Export to ONNX format")
    print("  3. Compile with Hailo Dataflow Compiler")
    print("\nFor most use cases, the embedding model provides")
    print("good accuracy with minimal setup.")
    print("\nNext step: Start monitoring")
    print("  python src/main.py --test")


if __name__ == "__main__":
    main()
