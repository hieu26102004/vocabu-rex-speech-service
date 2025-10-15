import whisper
import os

MODEL_NAMES = ["tiny", "base", "small", "medium", "large"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "whisper")
os.makedirs(MODEL_DIR, exist_ok=True)

def is_model_downloaded(model_name, model_dir):
    # Whisper lưu model ở model_dir/model_name.pt hoặc model_dir/model_name/model.bin
    pt_path = os.path.join(model_dir, f"{model_name}.pt")
    bin_path = os.path.join(model_dir, model_name, "model.bin")
    return os.path.exists(pt_path) or os.path.exists(bin_path)

def download_models():
    for name in MODEL_NAMES:
        if is_model_downloaded(name, MODEL_DIR):
            print(f"Model '{name}' already exists. Skipping download.")
            continue
        print(f"Downloading Whisper model: {name}")
        whisper.load_model(name, download_root=MODEL_DIR)
    print("All Whisper models downloaded.")

if __name__ == "__main__":
    download_models()
