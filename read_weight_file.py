import torch
from config import *
from evaluate import print_predicted_results

# Load checkpoint
state_dict = torch.load(
    "project/path/best_resnet18_epoch_99.pth", map_location="cpu")

best_model = MODEL_DICT["resnet18"]()
best_model.load_state_dict(state_dict)

best_model.eval()

# 3. Xem toàn bộ tên layer và weight shape
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")

# 4. Ví dụ: in ra 5 giá trị đầu tiên của mỗi weight
for name, param in state_dict.items():
    print(f"\nLayer: {name} | Shape: {param.shape}")
    print(param.view(-1)[:5])  # in 5 giá trị đầu tiên
