from huggingface_hub import snapshot_download, HfApi
import os

# Target Directory
target_dir = os.path.join(os.path.dirname(__file__), "data/motions")
os.makedirs(target_dir, exist_ok=True)

# Repository ID (Based on search)
repo_id = "lvhaidong/LAFAN1_Retargeting_Dataset"

print(f"Attempting to download {repo_id} to {target_dir}...")

try:
    # Try downloading specifically for H1 if folders exist
    # Pattern matching: usually datasets have 'h1' folder
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=["*H1*", "*h1*"], # Filter for H1 files
        local_dir_use_symlinks=False
    )
    print("Download complete!")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("Searching for similar repos...")
    api = HfApi()
    repos = api.list_datasets(search="LAFAN1")
    print("Found potential datasets:")
    for r in repos:
        print(f" - {r.id}")
