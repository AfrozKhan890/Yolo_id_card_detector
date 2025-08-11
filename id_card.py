import os
from ultralytics import YOLO
import torch

# # ✅ Check GPU Availability
# print("CUDA Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU Name:", torch.cuda.get_device_name(0))

# ✅ User input image path
image_path = input("Enter image path: ").strip()

# ✅ Model path
best_model_path = os.path.join(os.getcwd(), "best.pt")

# ✅ Output folder
output_dir = r"C:\Users\user\Desktop\Project"
os.makedirs(output_dir, exist_ok=True)

# ✅ Load model
model = YOLO(best_model_path)

# ✅ Predict and save to custom folder
results = model.predict(
    source=image_path,
    save=True,
    conf=0.3,
    project=output_dir,  # Save here
    name="",             # Avoid extra subfolder
    exist_ok=True        # Overwrite if exists
)

print(f"✅ Prediction Completed. Check here: {output_dir}")
