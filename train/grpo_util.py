import requests

try:
    response = requests.get("https://api.wandb.ai", timeout=5)
    if response.status_code == 200:
        print("可以访问 wandb 云端，已联网。")
    else:
        print("无法正常访问 wandb 云端，状态码：", response.status_code)
except Exception as e:
    print("无法访问 wandb 云端，错误：", e)