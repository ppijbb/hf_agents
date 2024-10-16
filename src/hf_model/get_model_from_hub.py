from huggingface_hub import snapshot_download

model_id = "Gunulhona/Gemma-Ko-Merge"

snapshot_download(
    repo_id=model_id, 
    local_dir="sample/ko", 
    local_dir_use_symlinks=False, 
    revision="main")

model_id = "Gunulhona/Gemma-Ko-Merge-PEFT"

# snapshot_download(
#     repo_id=model_id, 
#     local_dir="sample", 
#     local_dir_use_symlinks=False, 
#     revision="main")
