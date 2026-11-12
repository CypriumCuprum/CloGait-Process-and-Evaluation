import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import os

INPUT_ROOT = Path(os.environ.get("GAIT_DATA_ROOT", "./data")) / "person_frames_png"
OUTPUT_ROOT = Path(os.environ.get("GAIT_DATA_ROOT", "./data")) / "romp_smpl"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

ROMP_CMD = "romp"
GPU_ID = "0"
NUM_WORKERS = 1

TRACKING_FILE = OUTPUT_ROOT / "processed_romp.txt"
FAILED_FILE = OUTPUT_ROOT / "failed_romp.txt"
STILL_BAD_FILE = "./still_bad_after_reprocess.txt" 


def process_folder(view_folder: str):
    view_dir = Path(view_folder)

    try:
        person = view_dir.parents[1].name
        cloth = view_dir.parents[0].name
        view_name = view_dir.name

        save_dir = OUTPUT_ROOT / person / cloth / view_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            ROMP_CMD,
            "--mode=video",
            "--calc_smpl",
            f"-i={str(view_dir)}",
            f"-o={str(save_dir)}",
            "--GPU=0",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return ("fail", str(view_dir), result.stderr)

        npz_files = list(save_dir.glob("*.npz"))
        if not npz_files:
            return ("fail", str(view_dir), "ROMP không sinh ra file .npz nào.")

        for f in save_dir.iterdir():
            if f.suffix != ".npz":
                f.unlink() if f.is_file() else shutil.rmtree(f)

        video_results = save_dir / "video_results.npz"
        if video_results.exists():
            video_results.unlink()

        return ("ok", str(view_dir))

    except Exception as e:
        return ("fail", str(view_folder), str(e))


def has_bad_frames(view_dir):
    save_dir = OUTPUT_ROOT / view_dir.relative_to(INPUT_ROOT)

    for png in view_dir.glob("*.png"):
        npz_path = save_dir / f"{png.stem}.npz"

        if not npz_path.exists():
            return True

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "results" not in data.files:
                    return True
        except Exception:
            return True

    return False


def log_still_bad(view_dir: Path):
    """Ghi log những frame vẫn thiếu sau reprocess, không ảnh hưởng flow."""
    save_dir = OUTPUT_ROOT / view_dir.relative_to(INPUT_ROOT)
    rel = str(view_dir.relative_to(INPUT_ROOT))

    png_count = len(list(view_dir.glob("*.png")))
    npz_count = len(list(save_dir.glob("*.npz"))) if save_dir.exists() else 0

    missing_frames = []
    for png in sorted(view_dir.glob("*.png")):
        npz_path = save_dir / f"{png.stem}.npz"
        if not npz_path.exists():
            missing_frames.append(png.stem)

    if png_count != npz_count or missing_frames:
        with open(STILL_BAD_FILE, "a") as f:
            f.write(f"{rel} | png={png_count} npz={npz_count} | missing: {', '.join(missing_frames) if missing_frames else 'none (count mismatch)'}\n")
            f.flush()


def main():
    processed = set()
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, "r") as f:
            processed = set(line.strip() for line in f if line.strip())

    view_folders = sorted([
        p for p in INPUT_ROOT.glob("*/*/*")
        if p.is_dir() and p.name.startswith("View")
    ])

    to_process = []
    reprocess_set = set()  # 👈 track cái nào là reprocess

    for v in view_folders:
        rel = str(v.relative_to(INPUT_ROOT))

        if rel not in processed:
            to_process.append(v)
        elif has_bad_frames(v):
            print(f"Reprocess: {rel}")
            save_dir = OUTPUT_ROOT / rel
            for f in save_dir.glob("*.npz"):
                f.unlink()
            to_process.append(v)
            reprocess_set.add(str(v))  # 👈 đánh dấu

    print(f"Total view folders: {len(view_folders)}")
    print(f"Previously processed: {len(view_folders) - len(to_process)}")
    print(f"To process this run: {len(to_process)}")

    if not to_process:
        print("No new folders to process. Exiting!")
        return

    failed = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor, \
         open(TRACKING_FILE, "a") as track_f:

        futures = [executor.submit(process_folder, str(v)) for v in to_process]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            status, v_path = result[0], result[1]

            if status == "ok":
                rel = str(Path(v_path).relative_to(INPUT_ROOT))
                track_f.write(rel + "\n")
                track_f.flush()

                if v_path in reprocess_set:
                    log_still_bad(Path(v_path))

            elif status == "fail":
                failed.append(v_path)
                print(f"\nERROR: {v_path}\n{result[2]}")

    if failed:
        with open(FAILED_FILE, "w") as f:
            for item in failed:
                f.write(item + "\n")

    print(f"\nFinished. Success: {len(to_process) - len(failed)} | Failed: {len(failed)}")


if __name__ == "__main__":
    main()