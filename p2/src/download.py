from huggingface_hub import snapshot_download

# 使用你可访问的镜像（hf-mirror.com 最常见）
local_dir = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B-Base",
    repo_type="model",
    local_dir="qwen3_local",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com"
)

print("Model downloaded to:", local_dir)
