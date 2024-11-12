from pathlib import Path
from helper.preprocess_img import preprocess_image
from PIL import UnidentifiedImageError


def process_images(input_dir: Path, output_dir: Path):
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() in supported_extensions:
            try:
                processed_image = preprocess_image(str(image_path))

                if processed_image:
                    output_path = output_dir / image_path.name
                    processed_image.save(output_path)


            except UnidentifiedImageError:
                print(f"Unidentified image file: {image_path.name}")
            except Exception as e:
                print(f"Failed to process {image_path.name}: {e}")
        else:
            print(f"Unsupported file type skipped: {image_path.name}")

def main():
    input_directory = Path("/Users/adil/Desktop/Codes/atml_cv/large-receipt-image-dataset-SRD")
    output_directory = Path("/Users/adil/Desktop/Codes/atml_cv/processedimg")
    process_images(input_directory, output_directory)

if __name__ == "__main__":
    main()