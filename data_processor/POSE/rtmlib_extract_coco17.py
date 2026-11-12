#!/usr/bin/env python3
"""
Extract COCO 17-keypoint pose từ ảnh CROPPED bằng BATCH INFERENCE qua ONNXRuntime.
Tự động đồng bộ input size theo model ONNX thực tế.
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODE_MAP = {
    "rtmpose-s": "lightweight",
    "rtmpose-m": "balanced",
    "rtmpose-l": "performance",
}


class RTMPoseBatchExtractor:
    def __init__(self, model_name: str = "rtmpose-m", device: str = "cuda"):
        try:
            from rtmlib import Body
        except ImportError:
            raise ImportError("Vui lòng cài đặt: pip install rtmlib onnxruntime-gpu")

        mode = MODE_MAP.get(model_name, "balanced")
        log.info(f"Đang khởi tạo RTMPose [{model_name}] (mode='{mode}') trên {device.upper()}...")

        body_helper = Body(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device=device,
        )

        if hasattr(body_helper, "pose_model"):
            self.session = body_helper.pose_model.session
        elif hasattr(body_helper, "pose"):
            self.session = body_helper.pose.session
        else:
            raise AttributeError("Không tìm thấy ONNX session trong rtmlib.Body.")

        self.input_name = self.session.get_inputs()[0].name
        
        # Lấy kích thước input trực tiếp từ ONNX model (tránh lệch shape)
        # Shape thường là [Batch, Channel, Height, Width]
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 256
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 192
        log.info(f"Kích thước input chuẩn của model: Width={self.input_width}, Height={self.input_height}")

        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        log.info("Khởi tạo session ONNXRuntime thành công!")

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        target_w, target_h = self.input_width, self.input_height

        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        normalized = (canvas.astype(np.float32) - self.mean) / self.std
        tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
        return tensor, scale, dx, dy

    def decode_simcc(self, simcc_x: np.ndarray, simcc_y: np.ndarray, scales: List[float], pads: List[Tuple[int, int]]) -> np.ndarray:
        B, K, _ = simcc_x.shape
        kpts = np.zeros((B, K, 3), dtype=np.float32)

        for b in range(B):
            scale = scales[b]
            dx, dy = pads[b]

            x_locs = np.argmax(simcc_x[b], axis=-1) / 2.0
            y_locs = np.argmax(simcc_y[b], axis=-1) / 2.0

            x_scores = np.max(simcc_x[b], axis=-1)
            y_scores = np.max(simcc_y[b], axis=-1)
            scores = np.maximum(0.0, (x_scores + y_scores) / 2.0)

            kpts[b, :, 0] = (x_locs - dx) / scale
            kpts[b, :, 1] = (y_locs - dy) / scale
            kpts[b, :, 2] = scores

        return kpts

    def infer_batch(self, batch_imgs: List[np.ndarray]) -> np.ndarray:
        tensors, scales, pads = [], [], []

        for img in batch_imgs:
            tensor, scale, dx, dy = self.preprocess(img)
            tensors.append(tensor)
            scales.append(scale)
            pads.append((dx, dy))

        batch_tensor = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        simcc_x, simcc_y = outputs[0], outputs[1]

        return self.decode_simcc(simcc_x, simcc_y, scales, pads)


def extract_sequence(extractor: RTMPoseBatchExtractor, img_paths: List[Path], batch_size: int = 32) -> np.ndarray:
    if not img_paths:
        return None

    all_kpts = []
    for i in range(0, len(img_paths), batch_size):
        chunk_paths = img_paths[i:i + batch_size]
        batch_imgs = []

        for p in chunk_paths:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((extractor.input_height, extractor.input_width, 3), dtype=np.uint8)
            batch_imgs.append(img)

        kpts_chunk = extractor.infer_batch(batch_imgs)
        all_kpts.append(kpts_chunk)

    return np.concatenate(all_kpts, axis=0)


def process_dataset(input_root: Path, output_root: Path, device: str, ext: str, model_name: str, batch_size: int):
    extractor = RTMPoseBatchExtractor(model_name=model_name, device=device)

    tasks = []
    subjects = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not subjects:
        log.error(f"Không tìm thấy subject nào trong {input_root}")
        return

    for subj_dir in subjects:
        for cond_dir in sorted([d for d in subj_dir.iterdir() if d.is_dir()]):
            for view_dir in sorted([d for d in cond_dir.iterdir() if d.is_dir()]):
                rel = Path(subj_dir.name) / cond_dir.name / view_dir.name
                out_path = output_root / rel / "skeleton.pkl"
                tasks.append((view_dir, out_path, rel))

    log.info(f"Tổng số sequences: {len(tasks)} | Batch Size: {batch_size}")
    total = skipped = failed = 0

    for view_dir, out_path, rel in tqdm(tasks, desc="Batch Extracting Poses"):
        if out_path.exists():
            skipped += 1
            continue

        imgs = sorted(view_dir.glob(f"*.{ext}"))
        if not imgs:
            alt = "jpg" if ext == "png" else "png"
            imgs = sorted(view_dir.glob(f"*.{alt}"))

        if not imgs:
            failed += 1
            continue

        seq = extract_sequence(extractor, imgs, batch_size=batch_size)

        if seq is None or len(seq) == 0:
            failed += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(seq, f, protocol=pickle.HIGHEST_PROTOCOL)

        total += 1

    log.info("=" * 60)
    log.info(f"Hoàn thành: {total} sequences | {skipped} bỏ qua | {failed} lỗi")


def main():
    parser = argparse.ArgumentParser(description="Batch RTMPose Extractor for OpenGait")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--model", default="rtmpose-m", choices=["rtmpose-s", "rtmpose-m", "rtmpose-l"])
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    process_dataset(
        Path(args.input),
        Path(args.output),
        args.device,
        args.ext,
        args.model,
        args.batch_size,
    )


if __name__ == "__main__":
    main()