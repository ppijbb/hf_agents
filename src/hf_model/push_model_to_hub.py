import os
from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_file(
    path_or_fileobj="sample/Gemma-Ko-Merge.gguf",
    path_in_repo="Gemma-Ko-Merge.gguf",
    repo_id="Gunulhona/Gemma-Ko-Merge",
    repo_type="model",
)