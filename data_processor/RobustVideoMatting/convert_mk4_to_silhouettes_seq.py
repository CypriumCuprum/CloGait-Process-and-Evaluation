import os
import cv2
import torch
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
import threading
# Giới hạn tối đa 100 ảnh chờ lưu. Đầy là GPU phải dừng lại đợi.
MAX_QUEUE = 500 
queue_limiter = threading.Semaphore(MAX_QUEUE)

# Direct import works natively since the script is inside the RVM folder
try:
    from model import MattingNetwork
except ImportError:
    print("Import Error: MattingNetwork class not found. Ensure this script is inside the RobustVideoMatting directory.")
    import sys
    sys.exit(1)

def crop_silhouette(
        sil: NDArray[np.uint8], 
        resize1: int = 128, 
        resize2: int = 64, 
        pixel_threshold: int = 800
    ) -> NDArray[np.uint8]:
    
    y = sil.sum(axis=1)
    crop_y1 = (y > pixel_threshold).argmax(axis=0)
    crop_y2 = (y > pixel_threshold).cumsum(axis=0).argmax(axis=0)
    sil = sil[crop_y1:crop_y2 + 1, :]

    ratio = sil.shape[1] / sil.shape[0]
    sil = cv2.resize(sil, (int(resize1 * ratio), resize1), interpolation=cv2.INTER_CUBIC)

    x_cumsum = sil.sum(axis=0).cumsum()
    x_center = np.searchsorted(x_cumsum, sil.sum() / 2)

    half_width = resize1 // 2
    crop_x1 = x_center - half_width
    crop_x2 = x_center + half_width

    if crop_x1 <= 0 or crop_x2 >= sil.shape[1]:
        crop_x1 += half_width
        crop_x2 += half_width
        padding = np.zeros((sil.shape[0], half_width))
        sil = np.concatenate([padding, sil, padding], axis=1)

    sil = sil[:, crop_x1:crop_x2].astype(np.uint8)
    ratio = sil.shape[1] / sil.shape[0]
    sil = cv2.resize(sil, (int(resize2 * ratio), resize2), interpolation=cv2.INTER_CUBIC)
    return sil

def auto_downsample_ratio(h, w):
    """
    Tự động tính toán tỷ lệ downsample để cạnh dài nhất không vượt quá 512px.
    (Giữ nguyên chất lượng cho ảnh nhỏ)
    """
    return min(512 / max(h, w), 1.0)

def load_rvm_model(weights_path, device):
    if not os.path.exists(weights_path):
        print(f"Weights file '{weights_path}' not found.")
        return None
    print("Loading RVM model (ResNet50)...")
    model = MattingNetwork('resnet50').to(device).eval()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Model loaded successfully.")
    return model

class OpenGaitDataset(Dataset):
    def __init__(self, input_root, output_root, skip_dirs):
        self.samples = []
        input_path = Path(input_root)
        output_path = Path(output_root)
        
        view_dirs = [d for d in input_path.rglob('*') if d.is_dir() and d.name.startswith('View')]
        need_process_dirs = [d for d in view_dirs if d.parent.name not in skip_dirs]
        
        for view_dir in need_process_dirs:
            relative_path = view_dir.relative_to(input_path)
            frames_dir = output_path / relative_path
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            image_files = sorted(list(view_dir.glob('*.png')))
            for count, img_file in enumerate(image_files, start=1):
                out_file = frames_dir / f"frame{count:05d}.png"
                self.samples.append((str(img_file), str(out_file)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, out_path = self.samples[idx]
        frame = cv2.imread(img_path)
        
        if frame is None:
            return torch.empty(0), out_path
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        return tensor_img, out_path

# Hàm luồng chạy nền (Background thread) dùng để crop và lưu ảnh
def process_and_save(mask_array, path, threshold):
    try:
        final_sil = crop_silhouette(mask_array, pixel_threshold=threshold)
        cv2.imwrite(path, final_sil)
    except Exception as e:
        print(f"\n  Error cropping {path}: {e}")
    finally:
        # Bắt buộc phải có: Lưu xong 1 ảnh thì nhả 1 chỗ trống trong hàng đợi
        queue_limiter.release()

def process_pipeline(input_root, output_root, weights_path, pixel_threshold, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_rvm_model(weights_path, device)
    if model is None: 
        return

    skip_dir = [] 
    
    print("Scanning dataset... This might take a few seconds/minutes for 7000 folders.")
    dataset = OpenGaitDataset(input_root, output_root, skip_dir)
    print(f"Total frames to process: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Khởi tạo ThreadPool với số lượng luồng tương đương với num_workers
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

    with torch.no_grad():
        for tensor_img, out_path in tqdm(dataloader, desc="GPU Processing", dynamic_ncols=True):
            out_path = out_path[0] 
            
            if tensor_img.numel() == 0:
                print(f"\nError reading image for output: {out_path}")
                continue

            tensor_img = tensor_img.to(device, non_blocking=True)
            
            # Xóa sạch bộ nhớ RNN cho từng ảnh rời rạc
            rec = [None] * 4 

            # Lấy chiều cao (h) và chiều rộng (w) của ảnh hiện tại
            h, w = tensor_img.shape[2:] 
            # Tính ratio tự động
            ds_ratio = auto_downsample_ratio(h, w)

            # Đưa vào model
            fgr, pha, *rec = model(tensor_img, *rec, downsample_ratio=ds_ratio)
            
            # Kéo mask từ GPU về CPU
            alpha = pha[0, 0].cpu().numpy()
            silhouette_mask = (alpha * 255).astype(np.uint8)

            # Chờ nếu hàng đợi đang có 100 ảnh chưa lưu kịp
            queue_limiter.acquire() 
            
            # Đã có chỗ trống, ném cho luồng ngầm xử lý
            executor.submit(process_and_save, silhouette_mask, out_path, pixel_threshold)

    print("\nGPU processing finished. Waiting for background tasks to save the final images to disk...")
    # Chốt chặn: chờ các luồng lưu xong toàn bộ ảnh cuối cùng rồi mới kết thúc script
    executor.shutdown(wait=True)

if __name__ == "__main__":
    INPUT_DIR = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "person_frames_png")
    OUTPUT_DIR = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "rvm_silhouettes_64x64_png")
    WEIGHTS = "rvm_resnet50.pth"
    SIL_PIXEL_THRESHOLD = 150
    
    # Bạn có thể tăng lên 8 hoặc 16 tùy thuộc vào cấu hình CPU (Core/Thread)
    NUM_WORKERS = 4 
    
    process_pipeline(INPUT_DIR, OUTPUT_DIR, WEIGHTS, SIL_PIXEL_THRESHOLD, NUM_WORKERS)
    print("\nPipeline completed successfully.")