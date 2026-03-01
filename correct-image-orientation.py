from PIL import Image
from pathlib import Path
from check_orientation.pre_trained_models import create_model
import albumentations as albu
import torch
import numpy as np
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb

def rotate_in_place(image_path, angle):
  img = Image.open(image_path)

  rotated = img.rotate(angle, expand=True)

  # Overwrite original file
  rotated.save(image_path)

  img.close()

def fix_orientation(folder_path):
  model = create_model("swsl_resnext50_32x4d")
  model.eval()

  transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)

  image_folder = Path(folder_path)

  valid_extensions = ('.jpg', '.jpeg', '.png')

  print(f"Scanning for orientation issues in: {image_folder}\n")

  found_rotated = False
  count = 0
  # Loop through all files in the directory
  for file_path in image_folder.rglob('*'):
    if file_path.is_dir():
      continue

    if file_path.suffix.lower() in valid_extensions:
      try:
        # 1. Read and prepare the image
        image = load_rgb(file_path)
        temp = []
        for k in [0, 1, 2, 3]:
          x = transform(image=np.rot90(image, k))["image"]
          temp += [tensor_from_rgb_image(x)]
        with torch.no_grad():
          prediction = model(torch.stack(temp)).numpy()
        np.set_printoptions(precision=6, suppress=True)
        predicted_degree = np.argmax(prediction[0])*90
        # 3. Report only if rotation is needed
        if predicted_degree != 0:
          count = count+1
          rotate_in_place(file_path, 360-predicted_degree)
          found_rotated = True       
      except Exception as e:
        print(f"Error checking {file_path.name}: {e}")
  if not found_rotated:
    print("All images are currently straight (0 degrees).")
  else:
    print(f"Total images with abnormal orientation = {count}")

if __name__ == "__main__":
  fix_orientation("./compressed-photos/walima-cam2")