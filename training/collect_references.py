#!/usr/bin/env python3
"""
Reference Image Collector
==========================
Downloads reference images of the car make/model for training.

Uses public image searches to gather training data.
Images are used locally only for fine-tuning the car detector.

Run with: python training/collect_references.py
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

import yaml

logger = logging.getLogger(__name__)


def collect_reference_images(car_profile: dict, output_dir: Path, max_images: int = 50) -> List[str]:
    """
    Collect reference images for the car model.

    Args:
        car_profile: Car profile dict with make/model/year/colour
        output_dir: Directory to save images
        max_images: Maximum number of images to collect

    Returns:
        List of downloaded image paths
    """
    vehicle = car_profile.get("vehicle", {})

    make = vehicle.get("make", "")
    model = vehicle.get("model", "")
    year = vehicle.get("year", "")
    colour = vehicle.get("colour", {}).get("primary", "")
    body_style = vehicle.get("body_style", "")

    if not make or not model:
        logger.error("Car make and model are required in car_profile.yaml")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build search queries
    queries = build_search_queries(make, model, year, colour, body_style)

    print(f"\nCollecting reference images for: {year} {make} {model}")
    print(f"Colour: {colour or 'any'}")
    print(f"Body style: {body_style or 'any'}")
    print(f"Output directory: {output_dir}\n")

    downloaded = []

    for query in queries[:3]:  # Use top 3 queries
        print(f"Searching: {query}")
        images = search_and_download(query, output_dir, max_per_query=max_images // 3)
        downloaded.extend(images)

        if len(downloaded) >= max_images:
            break

        time.sleep(1)  # Rate limiting

    # Remove duplicates (by hash)
    unique_images = deduplicate_images(downloaded)

    print(f"\nCollected {len(unique_images)} unique reference images")

    # Save metadata
    save_collection_metadata(output_dir, car_profile, unique_images)

    return unique_images


def build_search_queries(
    make: str,
    model: str,
    year: Optional[int],
    colour: str,
    body_style: str
) -> List[str]:
    """Build search queries for image collection."""
    queries = []

    # Most specific query first
    base = f"{make} {model}"

    if year:
        base = f"{year} {base}"

    # Colour-specific query
    if colour:
        queries.append(f"{colour} {base}")

    # Body style query
    if body_style:
        queries.append(f"{base} {body_style}")

    # General query
    queries.append(base)

    # Add "car" for clarity
    queries.append(f"{base} car")

    # Different angles
    queries.append(f"{base} front view")
    queries.append(f"{base} side view")

    return queries


def search_and_download(query: str, output_dir: Path, max_per_query: int = 20) -> List[str]:
    """
    Search for images and download them.

    Note: This uses a simple approach with requests.
    For production, consider using an API like Bing Image Search.
    """
    downloaded = []

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("requests or beautifulsoup4 not installed")
        logger.info("Install with: pip install requests beautifulsoup4")
        return downloaded

    # Simple image search via DuckDuckGo
    # Note: This is for demonstration. For production use, consider proper APIs
    search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CarMonitor/1.0)"
    }

    try:
        # Note: DuckDuckGo uses JavaScript for image loading
        # This won't work directly - you'd need selenium or an API

        # Alternative: Use a placeholder approach for now
        logger.info(f"Query: {query}")
        logger.info("Note: Automated image download requires API access.")
        logger.info("Please manually download reference images or use an image search API.")

        # Create a placeholder file with instructions
        instructions_file = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, "w") as f:
            f.write("Reference Image Collection\n")
            f.write("=" * 40 + "\n\n")
            f.write("Please manually download reference images of your car.\n\n")
            f.write("Suggested search queries:\n")
            f.write(f"  - {query}\n")
            f.write("\nTips:\n")
            f.write("  - Download 20-50 images\n")
            f.write("  - Include various angles (front, side, rear)\n")
            f.write("  - Include similar lighting conditions to your camera\n")
            f.write("  - Match your car's colour if possible\n")
            f.write("  - Save images in JPG or PNG format\n")
            f.write(f"\nSave images to: {output_dir}\n")

    except Exception as e:
        logger.error(f"Search error: {e}")

    return downloaded


def download_image(url: str, output_path: Path) -> bool:
    """Download a single image."""
    try:
        import requests

        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False

        # Save image
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except Exception as e:
        logger.debug(f"Download failed for {url}: {e}")
        return False


def deduplicate_images(image_paths: List[str]) -> List[str]:
    """Remove duplicate images based on file hash."""
    seen_hashes = set()
    unique = []

    for path in image_paths:
        try:
            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique.append(path)
            else:
                # Remove duplicate
                os.remove(path)

        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")

    return unique


def save_collection_metadata(output_dir: Path, car_profile: dict, images: List[str]):
    """Save metadata about the collected images."""
    from datetime import datetime

    metadata = {
        "collection_date": datetime.now().isoformat(),
        "car_profile": {
            "make": car_profile.get("vehicle", {}).get("make"),
            "model": car_profile.get("vehicle", {}).get("model"),
            "year": car_profile.get("vehicle", {}).get("year"),
            "colour": car_profile.get("vehicle", {}).get("colour", {}).get("primary"),
        },
        "total_images": len(images),
        "images": [str(Path(p).name) for p in images]
    }

    metadata_path = output_dir / "collection_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Metadata saved to {metadata_path}")


def list_local_images(directory: Path) -> List[str]:
    """List all images in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []

    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))

    return [str(p) for p in images]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect reference images for car recognition")
    parser.add_argument("--profile", default="config/car_profile.yaml", help="Car profile file")
    parser.add_argument("--output", default="data/reference_images", help="Output directory")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum images to collect")
    parser.add_argument("--list-local", action="store_true", help="List existing local images")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    output_dir = Path(args.output)

    # Load car profile
    try:
        with open(args.profile, "r") as f:
            car_profile = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Car profile not found: {args.profile}")
        print("Please create config/car_profile.yaml with your car details.")
        sys.exit(1)

    if args.list_local:
        # Just list existing images
        output_dir.mkdir(parents=True, exist_ok=True)
        images = list_local_images(output_dir)
        print(f"\nFound {len(images)} local reference images:")
        for img in images[:20]:
            print(f"  {Path(img).name}")
        if len(images) > 20:
            print(f"  ... and {len(images) - 20} more")
        sys.exit(0)

    # Check car profile is filled in
    vehicle = car_profile.get("vehicle", {})
    if not vehicle.get("make") or not vehicle.get("model"):
        print("ERROR: Please fill in your car details in config/car_profile.yaml")
        print("Required: make and model")
        sys.exit(1)

    # Collect images
    images = collect_reference_images(car_profile, output_dir, args.max_images)

    # Also check for manually added images
    all_images = list_local_images(output_dir)

    print(f"\nTotal reference images available: {len(all_images)}")

    if len(all_images) < 10:
        print("\nRecommendation: Add more reference images for better recognition.")
        print(f"Place images in: {output_dir}")

    print("\nNext step: Run fine-tuning")
    print("  python training/fine_tune.py")


if __name__ == "__main__":
    main()
