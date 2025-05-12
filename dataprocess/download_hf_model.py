import argparse
import os
from huggingface_hub import snapshot_download

def download_model(model_name: str, target_path: str):
    print(f"准备下载模型：{model_name}")
    print(f"保存到路径：{target_path}")

    # 创建目标路径（如果不存在）
    os.makedirs(target_path, exist_ok=True)

    # 下载整个模型快照
    snapshot_download(
        repo_id=model_name,
        local_dir=target_path,
        local_dir_use_symlinks=False  # 避免软链接，直接拷贝文件
    )

    print(f"模型 {model_name} 下载完成，保存在 {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型")
    parser.add_argument("--model", type=str, required=True, help="模型名称，例如：bert-base-uncased")
    parser.add_argument("--path", type=str, required=True, help="保存模型的本地路径")

    args = parser.parse_args()

    download_model(args.model, args.path)
