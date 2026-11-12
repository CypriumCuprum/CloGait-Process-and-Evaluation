import os
import subprocess
from pathlib import Path

def convert_mkv_to_mp4_tree(input_root, output_root):
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    mkv_files = list(input_path.rglob('*.mkv'))
    
    if not mkv_files:
        print("No .mkv files found.")
        return

    for mkv_file in mkv_files:
        relative_path = mkv_file.relative_to(input_path)
        mp4_file = output_path / relative_path.with_suffix('.mp4')
        
        mp4_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting: {relative_path} -> {mp4_file.name}")
        
        command = ['ffmpeg', '-i', str(mkv_file), '-codec', 'copy', str(mp4_file), '-y']
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            fallback_command = ['ffmpeg', '-i', str(mkv_file), str(mp4_file), '-y']
            try:
                subprocess.run(fallback_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except Exception as e:
                print(f"Error converting {mkv_file.name}: {e}")

if __name__ == "__main__":
    INPUT_ROOT = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "original_mkv")
    OUTPUT_ROOT = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "mp4video")
    
    convert_mkv_to_mp4_tree(INPUT_ROOT, OUTPUT_ROOT)
    print("Completed!")