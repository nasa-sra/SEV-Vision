from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def convert_depth_to_model_range(
	depth_raw: np.ndarray,
	depth_scale_m_per_unit: float,
	max_depth_m: float,
) -> np.ndarray:
	"""Convert raw RealSense depth units to model range [0, 1].

	Output convention matches the project:
	- 0.0 -> 0 meters
	- 1.0 -> max_depth_m (and invalid pixels)
	"""
	depth_m = depth_raw.astype(np.float32) * depth_scale_m_per_unit
	depth_m = np.clip(depth_m, 0.0, max_depth_m)
	depth_norm = depth_m / max_depth_m

	# RealSense uses 0 for invalid depth in z16 streams.
	invalid_mask = depth_raw == 0
	depth_norm[invalid_mask] = 1.0

	return depth_norm


def load_realsense_raw(
	input_path: Path,
	width: int,
	height: int,
	endian: str,
) -> np.ndarray:
	"""Load RealSense RAW depth bytes as a 2D uint16 array."""
	byte_order = "<" if endian == "little" else ">"
	dtype = np.dtype(f"{byte_order}u2")

	raw = np.fromfile(input_path, dtype=dtype)
	expected = width * height
	if raw.size != expected:
		raise ValueError(
			f"Unexpected element count in {input_path}. "
			f"Got {raw.size}, expected {expected} for {width}x{height}."
		)

	return raw.reshape((height, width))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert Intel RealSense RAW depth to model-normalized [0,1] depth."
	)
	parser.add_argument("input_raw", type=Path, help="Path to input .raw depth file")
	parser.add_argument(
		"--width",
		type=int,
		default=848,
		help="Frame width in pixels (D455 default stream often 848)",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=480,
		help="Frame height in pixels (D455 default stream often 480)",
	)
	parser.add_argument(
		"--depth-scale",
		type=float,
		default=0.001,
		help="Meters per raw depth unit (D455 default typically 0.001)",
	)
	parser.add_argument(
		"--max-depth-m",
		type=float,
		default=80.0,
		help="Depth corresponding to 1.0 in normalized output",
	)
	parser.add_argument(
		"--endian",
		choices=["little", "big"],
		default="little",
		help="Byte order of the RAW file",
	)
	parser.add_argument(
		"--output-npy",
		type=Path,
		default=None,
		help="Optional output .npy path (defaults to input name + _norm.npy)",
	)
	parser.add_argument(
		"--output-png",
		type=Path,
		default=None,
		help="Optional preview PNG path (8-bit visualization only)",
	)
	args = parser.parse_args()

	input_path: Path = args.input_raw
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	depth_raw = load_realsense_raw(
		input_path=input_path,
		width=args.width,
		height=args.height,
		endian=args.endian,
	)

	depth_norm = convert_depth_to_model_range(
		depth_raw=depth_raw,
		depth_scale_m_per_unit=args.depth_scale,
		max_depth_m=args.max_depth_m,
	)

	output_npy = args.output_npy or input_path.with_name(f"{input_path.stem}_norm.npy")
	np.save(output_npy, depth_norm)

	print(f"Input:       {input_path}")
	print(f"Resolution:  {args.width}x{args.height}")
	print(f"Depth scale: {args.depth_scale} m/unit")
	print(f"Saved NPY:   {output_npy}")
	print(f"Range:       min={depth_norm.min():.6f}, max={depth_norm.max():.6f}")
	print(f"Invalid px:  {int((depth_raw == 0).sum())}")

	if args.output_png is not None:
		preview = (depth_norm * 255.0).round().astype(np.uint8)
		ok = cv2.imwrite(str(args.output_png), preview)
		if not ok:
			raise ValueError(f"Failed to write preview PNG: {args.output_png}")
		print(f"Saved PNG:   {args.output_png}")


if __name__ == "__main__":
	main()
