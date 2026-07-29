import argparse
import gc
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from pathlib import Path
from segformer_realtime import TorchSegformerBackend, TrtSegformerBackend

# Depth range used for label validity (adjust in code as needed).
DEPTH_LABEL_MIN_M = 3.0
DEPTH_LABEL_MAX_M = 80.0
ROAD_OVERRIDE_CONFIDENCE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime SegFormer road segmentation with Torch or TensorRT backend.")
    parser.add_argument(
        "--backend",
        choices=["torch", "trt"],
        default="torch",
        help="Inference backend.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("segformer_depth_finetuned_17"),
        help="Path used for preprocessor and label config.",
    )
    parser.add_argument(
        "--trt-engine",
        type=Path,
        default=Path("trt/segformer_depth_finetuned_trt.engine"),
        help="TensorRT engine path (used when --backend trt).",
    )
    parser.add_argument(
        "--realsense-bag",
        type=Path,
        default=Path("video/Building9ToRockyard.db3"),
        help="RealSense recording file.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print rolling per-stage timing breakdown.",
    )
    parser.add_argument(
        "--timing-window",
        type=int,
        default=120,
        help="Number of frames per timing report.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip segmentation/depth rendering and display only FPS.",
    )
    parser.add_argument(
        "--display-drivable",
        action="store_true",
        help=(
            "Highlight only non-drivable areas in red. Drivable/sky/out-of-range stays uncolored."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = rs.config()
    cfg.enable_device_from_file(str(args.realsense_bag), repeat_playback=False)

    # if not args.model_path.exists():
    #     raise FileNotFoundError(f"Model folder not found: {args.model_path}")

    backend = None
    pipe = None
    profile = None
    spatial_filter = None
    temporal_filter = None
    hole_filling = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    if args.backend == "torch":
        backend = TorchSegformerBackend(args.model_path, device=device, use_fp16=use_fp16)
    else:
        backend = TrtSegformerBackend(args.model_path, args.trt_engine, enable_timing=args.timing)

    id2label = backend.id2label
    num_classes = backend.num_classes

    road_class_ids = [
        int(class_id) for class_id, name in id2label.items()
        if "road" in str(name).lower()
    ]
    undefined_class_ids = [
        int(class_id) for class_id, name in id2label.items()
        if any(
            token in str(name).lower()
            for token in ("undefined", "unlabeled", "unknown", "void")
        )
    ]
    sky_class_ids = [
        int(class_id) for class_id, name in id2label.items()
        if "sky" in str(name).lower()
    ]
    building_class_ids = [
        int(class_id) for class_id, name in id2label.items()
        if "building" in str(name).lower()
    ]

    drivable_class_ids = set(road_class_ids) | set(undefined_class_ids) | set(sky_class_ids) | set(building_class_ids)

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        palette[class_id] = [
            (37 * class_id) % 256,
            (67 * class_id + 53) % 256,
            (97 * class_id + 101) % 256,
        ]
    for class_id in road_class_ids:
        if 0 <= class_id < num_classes:
            palette[class_id] = [255, 0, 0]

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_m = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale_m} meters/unit")
    print(f"Using backend: {args.backend}")
    print(f"Label depth range: {DEPTH_LABEL_MIN_M:.1f}m to {DEPTH_LABEL_MAX_M:.1f}m")
    if args.backend == "trt":
        preferred_hw = backend.get_preferred_hw()
        if preferred_hw is not None:
            print(f"TensorRT profile opt input shape (H, W): {preferred_hw}")
    print("Depth post-processing: spatial + temporal + hole-filling")
    if args.no_render:
        print("No-render mode enabled (FPS-only display).")
    if args.display_drivable:
        print("Display-drivable mode enabled (red=non-drivable only; road/sky/undefined uncolored).")
    if args.timing:
        print(f"Timing enabled (window={args.timing_window} frames)")

    playback = profile.get_device().as_playback()
    playback.set_real_time(True)

    prev_time = None
    fps = 0.0
    max_depth_m = 80.0
    previous_logits = None
    alpha = 0.15
    timing_enabled = args.timing
    fps_canvas = np.zeros((180, 480, 3), dtype=np.uint8)

    timing_stats = {
        "preprocess": 0.0,
        "inference": 0.0,
        "infer_enqueue": 0.0,
        "infer_sync": 0.0,
        "postprocess": 0.0,
        "render": 0.0,
        "total": 0.0,
    }
    timing_count = 0

    def maybe_sync_cuda() -> None:
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        while True:
            frame_t0 = time.perf_counter() if timing_enabled else 0.0
            try:
                frames = pipe.wait_for_frames()
            except RuntimeError:
                break

            color = frames.get_color_frame()
            if not color:
                continue

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # depth_filtered = spatial_filter.process(depth_frame)
            # depth_filtered = temporal_filter.process(depth_filtered)
            # depth_filtered = hole_filling.process(depth_filtered)
            # depth_frame = depth_filtered.as_depth_frame()
            if not depth_frame:
                continue

            color_image = np.asanyarray(color.get_data())
            color_rgb = color_image

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_raw_display = depth_raw

            out_h, out_w = color_rgb.shape[0], color_rgb.shape[1]
            depth_m = depth_raw.astype(np.float32) * depth_scale_m
            """
            valid_depth_mask = (
                (depth_m >= DEPTH_LABEL_MIN_M)
                & (depth_m <= DEPTH_LABEL_MAX_M)
            )
            if valid_depth_mask.shape != (out_h, out_w):
                valid_depth_mask = cv2.resize(
                    valid_depth_mask.astype(np.uint8),
                    (out_w, out_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            """
            if timing_enabled:
                stage_t0 = time.perf_counter()

            prepared = backend.preprocess(
                color_rgb=color_rgb,
                depth_raw=depth_raw,
                depth_scale_m=depth_scale_m,
                max_depth_m=max_depth_m,
            )
            if timing_enabled:
                timing_stats["preprocess"] += time.perf_counter() - stage_t0

            if timing_enabled:
                maybe_sync_cuda()
                stage_t0 = time.perf_counter()
                logits = backend.infer(prepared)
                infer_enqueue_dt = time.perf_counter() - stage_t0
                sync_t0 = time.perf_counter()
                maybe_sync_cuda()
                infer_sync_dt = time.perf_counter() - sync_t0
                timing_stats["infer_enqueue"] += infer_enqueue_dt
                timing_stats["infer_sync"] += infer_sync_dt
                timing_stats["inference"] += time.perf_counter() - stage_t0
            else:
                logits = backend.infer(prepared)

            if timing_enabled:
                maybe_sync_cuda()
                stage_t0 = time.perf_counter()
            if args.no_render:
                prediction = None
            else:
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                )

                if previous_logits is None:
                    smoothed_logits = upsampled_logits
                else:
                    # smoothed_logits = alpha * upsampled_logits + (1 - alpha) * previous_logits
                    smoothed_logits = upsampled_logits
                previous_logits = smoothed_logits

                prediction_tensor = smoothed_logits.argmax(dim=1)
                
                if road_class_ids:
                    probabilities = torch.softmax(smoothed_logits, dim=1)
                    road_probabilities = probabilities[:, road_class_ids, :, :]
                    road_confidence, road_choice_local = torch.max(road_probabilities, dim=1)
                    road_class_tensor = torch.as_tensor(
                        road_class_ids,
                        device=smoothed_logits.device,
                        dtype=torch.long,
                    )
                    road_choice_global = road_class_tensor[road_choice_local]
                    road_override_mask = road_confidence > ROAD_OVERRIDE_CONFIDENCE
                    prediction_tensor = torch.where(road_override_mask, road_choice_global, prediction_tensor)
                

                prediction = prediction_tensor[0].detach().cpu().numpy().astype(np.int16)
                # prediction[~valid_depth_mask] = -1
            if timing_enabled:
                maybe_sync_cuda()
                timing_stats["postprocess"] += time.perf_counter() - stage_t0

            if timing_enabled:
                stage_t0 = time.perf_counter()
            now = time.perf_counter()
            if prev_time is not None:
                dt = now - prev_time
                if dt > 0.0:
                    current_fps = 1.0 / dt
                    fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)
            prev_time = now

            if args.no_render:
                fps_canvas.fill(0)
                cv2.putText(
                    fps_canvas,
                    "No-render benchmark",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (170, 170, 170),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    fps_canvas,
                    f"FPS: {fps:5.1f}",
                    (12, 126),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Segformer FPS", fps_canvas)
            else:
                if args.display_drivable:
                    labeled_mask = prediction >= 0
                    known_mask = labeled_mask & (prediction < num_classes)
                    drivable_mask = np.isin(prediction, list(drivable_class_ids))
                    # For labeled pixels that have unknown ids, treat as drivable by default.
                    drivable_mask = drivable_mask | (labeled_mask & (~known_mask))
                    non_drivable_mask = labeled_mask & (~drivable_mask)

                    overlay_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    overlay_rgb[non_drivable_mask] = [255, 0, 0]
                    blended_rgb = color_rgb.copy()
                    blended_rgb[non_drivable_mask] = (
                        color_rgb[non_drivable_mask] * 0.55 + overlay_rgb[non_drivable_mask] * 0.45
                    ).astype(np.uint8)
                    # Crop out the hood of the car (bottom third of the image)
                    class_color_rgb_cropped = blended_rgb[:out_h * 2 // 3, :, :]
                    blended_bgr = cv2.cvtColor(class_color_rgb_cropped, cv2.COLOR_RGB2BGR)

                    cv2.putText(
                        blended_bgr,
                        f"FPS: {fps:5.1f}",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        blended_bgr,
                        "Red: non-drivable only, No tint: drivable/sky/out-of-range depth",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Segformer Drivable", blended_bgr)
                else:
                    labeled_mask = prediction >= 0
                    overlay_rgb = np.zeros_like(color_rgb)
                    overlay_rgb[labeled_mask] = palette[prediction[labeled_mask].astype(np.int32)]
                    blended_rgb = color_rgb.copy()
                    blended_rgb[labeled_mask] = (
                        color_rgb[labeled_mask] * 0.55 + overlay_rgb[labeled_mask] * 0.45
                    ).astype(np.uint8)
                    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

                    sidebar_width = 300
                    sidebar = np.full((blended_bgr.shape[0], sidebar_width, 3), 24, dtype=np.uint8)
                    cv2.putText(
                        sidebar,
                        "Legend (visible classes)",
                        (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )

                    present_class_ids = np.unique(prediction[labeled_mask]) if np.any(labeled_mask) else np.array([], dtype=np.int16)
                    row_h = 22
                    y0 = 52
                    max_rows = max(1, (sidebar.shape[0] - y0 - 8) // row_h)
                    shown_ids = present_class_ids[:max_rows]

                    for row, class_id in enumerate(shown_ids):
                        y = y0 + row * row_h
                        class_id_int = int(class_id)
                        class_color_rgb = palette[class_id_int]
                        color_bgr = (int(class_color_rgb[2]), int(class_color_rgb[1]), int(class_color_rgb[0]))
                        cv2.rectangle(sidebar, (12, y - 12), (32, y + 8), color_bgr, thickness=-1)
                        cv2.rectangle(sidebar, (12, y - 12), (32, y + 8), (200, 200, 200), thickness=1)

                        class_name = id2label.get(class_id_int, f"class_{class_id_int}")
                        label_text = f"{class_id_int:2d}: {class_name}"
                        if len(label_text) > 37:
                            label_text = label_text[:34] + "..."

                        cv2.putText(
                            sidebar,
                            label_text,
                            (40, y + 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (230, 230, 230),
                            1,
                            cv2.LINE_AA,
                        )

                    hidden = int(present_class_ids.size - shown_ids.size)
                    if hidden > 0:
                        cv2.putText(
                            sidebar,
                            f"+ {hidden} more",
                            (12, sidebar.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )

                    cv2.putText(
                        blended_bgr,
                        f"FPS: {fps:5.1f}",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    overlay_with_sidebar = np.hstack([blended_bgr, sidebar])
                    cv2.imshow("Segformer Overlay", overlay_with_sidebar)

            if timing_enabled:
                timing_stats["render"] += time.perf_counter() - stage_t0
                timing_stats["total"] += time.perf_counter() - frame_t0
                timing_count += 1

            if timing_enabled and timing_count >= max(1, args.timing_window):
                to_ms = 1000.0 / timing_count
                pre_ms = timing_stats["preprocess"] * to_ms
                inf_ms = timing_stats["inference"] * to_ms
                inf_enqueue_ms = timing_stats["infer_enqueue"] * to_ms
                inf_sync_ms = timing_stats["infer_sync"] * to_ms
                post_ms = timing_stats["postprocess"] * to_ms
                ren_ms = timing_stats["render"] * to_ms
                total_ms = timing_stats["total"] * to_ms
                est_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
                print(
                    "Timing avg "
                    f"(n={timing_count}): preprocess={pre_ms:.2f} ms, "
                    f"infer={inf_ms:.2f} ms (enqueue={inf_enqueue_ms:.2f} ms, sync={inf_sync_ms:.2f} ms), "
                    f"post={post_ms:.2f} ms, "
                    f"render={ren_ms:.2f} ms, total={total_ms:.2f} ms "
                    f"(~{est_fps:.2f} FPS)"
                )

                if args.backend == "trt" and hasattr(backend, "consume_preprocess_timing"):
                    trt_pre = backend.consume_preprocess_timing()
                    trt_count = int(trt_pre.get("count", 0.0))
                    if trt_count > 0:
                        trt_scale = 1000.0 / trt_count
                        rgb_resize_ms = trt_pre["rgb_resize"] * trt_scale
                        rgb_to_float_ms = trt_pre["rgb_to_float"] * trt_scale
                        rgb_norm_ms = trt_pre["rgb_normalize"] * trt_scale
                        rgb_layout_ms = trt_pre["rgb_layout"] * trt_scale
                        depth_resize_ms = trt_pre["depth_resize"] * trt_scale
                        depth_encode_ms = trt_pre["depth_encode"] * trt_scale
                        pack_input_ms = trt_pre["pack_input"] * trt_scale
                        total_pre_ms = trt_pre["total"] * trt_scale
                        print(
                            "TRT preprocess breakdown "
                            f"(n={trt_count}): rgb_resize={rgb_resize_ms:.2f} ms, "
                            f"rgb_to_float={rgb_to_float_ms:.2f} ms, "
                            f"rgb_normalize={rgb_norm_ms:.2f} ms, "
                            f"rgb_layout={rgb_layout_ms:.2f} ms, "
                            f"depth_resize={depth_resize_ms:.2f} ms, "
                            f"depth_encode={depth_encode_ms:.2f} ms, "
                            f"pack_input={pack_input_ms:.2f} ms, "
                            f"total={total_pre_ms:.2f} ms"
                        )

                if args.backend == "trt" and hasattr(backend, "consume_infer_timing"):
                    trt_inf = backend.consume_infer_timing()
                    trt_count = int(trt_inf.get("count", 0.0))
                    if trt_count > 0:
                        trt_scale = 1000.0 / trt_count
                        host_cast_ms = trt_inf["host_cast"] * trt_scale
                        set_shape_ms = trt_inf["set_shape"] * trt_scale
                        alloc_ms = trt_inf["alloc"] * trt_scale
                        h2d_copy_ms = trt_inf["h2d_copy"] * trt_scale
                        execute_ms = trt_inf["execute"] * trt_scale
                        wait_stream_ms = trt_inf["wait_stream"] * trt_scale
                        total_inf_ms = trt_inf["total"] * trt_scale
                        print(
                            "TRT infer breakdown "
                            f"(n={trt_count}): host_cast={host_cast_ms:.2f} ms, "
                            f"set_shape={set_shape_ms:.2f} ms, "
                            f"alloc={alloc_ms:.2f} ms, "
                            f"h2d_copy={h2d_copy_ms:.2f} ms, "
                            f"execute={execute_ms:.2f} ms, "
                            f"wait_stream={wait_stream_ms:.2f} ms, "
                            f"total={total_inf_ms:.2f} ms"
                        )

                for key in timing_stats:
                    timing_stats[key] = 0.0
                timing_count = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cleanup_errors = []
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                cleanup_errors.append(f"backend close: {exc}")
        if pipe is not None:
            try:
                pipe.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                cleanup_errors.append(f"pipeline stop: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            cleanup_errors.append(f"window cleanup: {exc}")
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            cleanup_errors.append(f"CUDA cleanup: {exc}")

        backend = None
        pipe = None
        profile = None
        spatial_filter = None
        temporal_filter = None
        hole_filling = None
        gc.collect()

        if cleanup_errors:
            print("Cleanup warnings:")
            for err in cleanup_errors:
                print(f"  - {err}")


if __name__ == "__main__":
    main()