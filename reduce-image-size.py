from pathlib import Path
from PIL import Image

def compress_images(input_dir, output_dir):
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Loop through all files in the input folder
    print(f"Starting image compression from {input_folder}...")

    for file_path in input_folder.rglob('*'):
        if file_path.is_dir():
            continue

        if file_path.suffix.lower() in valid_extensions:

            relative_path = file_path.relative_to(input_folder)
            save_path = output_folder / relative_path

            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.exists():
                print(f"Skipping: {relative_path} (Already exists!)")
                continue

            try:
                with Image.open(file_path) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(
                        save_path,
                        quality=65,
                        optimize=True,
                        progressive=True
                    )
                    print(f"Compressed and saved: {relative_path}")
            except Exception as e:
                print(f"Could not process {relative_path}: {e}")
    print(f"\nDone! All files processed.")


if __name__ == "__main__":
    compress_images("./test-images", "./compressed-photos/test-images")