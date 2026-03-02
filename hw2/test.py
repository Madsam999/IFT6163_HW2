import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")