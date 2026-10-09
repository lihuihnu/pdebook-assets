"""Prepare the exact pinned CC0 camera fixture before any PDE experiment.

This script runs in an isolated scikit-image==0.25.2 extraction environment.
It never advances the PDE; it stops if license claim/version/image bytes disagree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import skimage
from skimage import data
from PIL import Image


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def prepare(cfg_path: Path, fixture_dir: Path, provenance_path: Path) -> None:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    src_cfg = cfg["source"]
    if (
        cfg["experiment_id"] != "ET004-05"
        or src_cfg["package"] != "scikit-image"
        or src_cfg["version"] != "0.25.2"
        or src_cfg["function"] != "skimage.data.camera"
        or src_cfg["raw_asset"] != "camera.png"
        or src_cfg["license"] != "CC0"
        or src_cfg["photographer"] != "Lav Varshney"
        or src_cfg["raw_shape"] != [512, 512]
        or src_cfg["raw_dtype"] != "uint8"
        or src_cfg["crop_rows"] != [128, 384]
        or src_cfg["crop_cols"] != [128, 384]
        or src_cfg["crop_shape"] != [256, 256]
    ):
        raise ValueError("the preregistered camera provenance contract changed")
    installed = importlib.metadata.version("scikit-image")
    if installed != "0.25.2" or skimage.__version__ != "0.25.2":
        raise RuntimeError(f"must use precisely scikit-image==0.25.2, got {installed}")
    documentation = inspect.getdoc(data.camera) or ""
    if "CC0" not in documentation or "Lav Varshney" not in documentation:
        raise PermissionError("pinned data.camera documentation lacks required CC0/photographer statement")
    if "scikit-image.org" not in src_cfg["doc_url"] or "/0.25.x/" not in src_cfg["doc_url"]:
        raise PermissionError("unrecognized pinned license-evidence URL")
    original = data.camera()
    if original.dtype != np.uint8 or original.shape != (512, 512):
        raise ValueError("unexpected original camera shape/dtype; STOP for scientific amendment")
    if not original.flags.c_contiguous or not np.isfinite(original).all():
        raise AssertionError("unexpected original camera memory/values")
    bundled_png = Path(data.data_dir) / "camera.png"
    if not bundled_png.is_file():
        raise FileNotFoundError("pinned scikit-image distribution camera.png not found")
    with Image.open(bundled_png) as im:
        file_mode = im.mode
        decoded = np.asarray(im).copy()
    if decoded.dtype != np.uint8 or decoded.shape != original.shape:
        raise AssertionError("the bundled camera.png is not a grayscale uint8 512x512 image")
    if not np.array_equal(decoded, original):
        raise AssertionError("data.camera() pixels differ from source package camera.png")
    crop_uint8 = np.ascontiguousarray(original[128:384,128:384])
    if crop_uint8.shape != (256, 256) or crop_uint8.dtype != np.uint8:
        raise AssertionError("fixed crop was altered")
    clean = crop_uint8.astype(np.float64) / 255.0
    if not np.all((clean >= 0) & (clean <= 1)):
        raise AssertionError("raw image outside normalized brightness range")
    origin_name = Path(src_cfg["original_png"]).name
    crop_name = Path(src_cfg["cropped_png"]).name
    if fixture_dir.exists() and any(fixture_dir.iterdir()):
        raise FileExistsError("refusing to overwrite preexisting camera fixture directory")
    if provenance_path.exists():
        raise FileExistsError("refusing to overwrite preexisting camera source provenance")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    original_destination = fixture_dir / origin_name
    crop_destination = fixture_dir / crop_name
    shutil.copyfile(bundled_png, original_destination)
    Image.fromarray(crop_uint8, mode="L").save(crop_destination, format="PNG")
    with Image.open(crop_destination) as check_png:
        saved_crop = np.asarray(check_png).copy()
    if not np.array_equal(saved_crop, crop_uint8):
        raise AssertionError("published cropped PNG changed any pixels")
    provenance = {
      "experiment_id": "ET004-05",
      "license_gate": "PASS",
      "source": {
        "package": "scikit-image", "version": installed,
        "data_function": "skimage.data.camera()",
        "package_module_path": str(Path(skimage.__file__).resolve()),
        "package_camera_asset_path": str(bundled_png.resolve()),
        "source_release_url": src_cfg["release_tag_url"],
        "source_asset_url": "https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/camera.png",
        "license_evidence_url": src_cfg["doc_url"],
        "license": "CC0", "photographer": "Lav Varshney",
        "camera_function_docstring_sha256": hashlib.sha256(documentation.encode("utf-8")).hexdigest()
      },
      "original": {
        "name": origin_name, "shape": list(original.shape),
        "dtype": str(original.dtype), "pil_mode": file_mode,
        "raw_png_sha256": sha_file(bundled_png),
        "raw_png_bytes": bundled_png.stat().st_size,
        "uint8_pixel_sha256": sha_array(original)
      },
      "crop": {
        "name": crop_name, "rows": [128,384], "cols": [128,384],
        "shape": [256,256], "dtype": "uint8",
        "png_sha256": sha_file(crop_destination),
        "png_bytes": crop_destination.stat().st_size,
        "uint8_pixel_sha256": sha_array(crop_uint8),
        "float64_div255_pixel_sha256": sha_array(clean)
      },
      "fixture_extraction_runtime": {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scikit_image": installed,
        "pillow": importlib.metadata.version("Pillow")
      },
      "license_boundary": "real CC0 photograph used as reference for ADDITIONAL SYNTHETIC Gaussian noise; source is not physical noise-free truth",
      "config_sha256": sha_file(cfg_path)
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("ET004-05 PHOTO_PROVENANCE PASS source_pkg=0.25.2 author=Lav_Varshney license=CC0")
    print("ET004-05 ORIGINAL_PNG_SHA256="+provenance["original"]["raw_png_sha256"])
    print("ET004-05 ORIGINAL_PIXELS_SHA256="+provenance["original"]["uint8_pixel_sha256"])
    print("ET004-05 CROP_PNG_SHA256="+provenance["crop"]["png_sha256"])
    print("ET004-05 CROP_FLOAT64_SHA256="+provenance["crop"]["float64_div255_pixel_sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--fixture-dir",type=Path,required=True)
    ap.add_argument("--provenance",type=Path,required=True)
    args=ap.parse_args()
    prepare(args.config,args.fixture_dir,args.provenance)
