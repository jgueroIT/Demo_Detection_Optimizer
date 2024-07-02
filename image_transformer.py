import os
import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

def clear_output_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            for root, dirs, files in os.walk(item_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting the file {file_path}. Reason: {e}")
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        print(f"Error deleting the directory {dir_path}. Reason: {e}")
            try:
                os.rmdir(item_path)
            except Exception as e:
                print(f"Error deleting the directory {item_path}. Reason: {e}")

def basicLinearTransform(img, alpha, beta):
    """Apply basic linear transformation for contrast and brightness."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gammaCorrection(img, gamma):
    """Apply gamma correction to the image."""
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

def save_image(image, path):
    """Save an image using PIL:
    We convert the OpenCV image (BGR) to RGB before saving it as a PIL image. This allows the images to be read in RGB directly with the file path in the main algorithm.
    This conversion is necessary because YOLO detects images loaded with OpenCV as BGR, and working with them in RGB would cause color conflicts.
    Therefore, we save the image in RGB and PIL format to ensure it is loaded afterward directly as RGB using its file path."""
    pil_image = Image.fromarray(image)
    pil_image.save(path)

def process_and_save_images(img_original, output_subfolder, output_filename, N, alpha_range, beta_range, gamma_range, transformation):
    """Process and save N transformed versions of the original image."""
    version_counter = 0
    for _ in range(N):
        alpha, beta, gamma = 1.0, 0.0, 1.0
        
        if 'B' in transformation:
            beta = np.random.choice(beta_range)
        if 'C' in transformation:
            alpha = np.random.choice(alpha_range)
        if 'G' in transformation:
            gamma = np.random.choice(gamma_range)

        img_corrected = basicLinearTransform(img_original, alpha, beta)
        img_gamma_corrected = gammaCorrection(img_corrected, gamma)
        img_gamma_corrected_rgb = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2RGB)

        result_filename = f"{output_filename}_v{version_counter}_ALTmodif_a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}.jpg"
        output_path = os.path.join(output_subfolder, result_filename)
        save_image(img_gamma_corrected_rgb, output_path)
        print(f"Randomly transformed image (Version {version_counter}): {result_filename}")

        version_counter += 1

def transform_image(input_image_path, output_folder, amplitude, repetitions, transformation, clean_folder):

    alpha_range = np.arange(0.75, 2.8, 0.05)  # CONTRAST. alpha=1 does not modify the image
    beta_range = np.arange(-60, 60, 1)  # BRIGHTNESS. beta=0 does not modify the image
    gamma_range = np.arange(0.5, 3.5, 0.05)  # GAMMA. gamma=1 does not modify the image

    if clean_folder:
        clear_output_folder(output_folder)

    if input_image_path.endswith('.jpg') or input_image_path.endswith('.png'):
        img_original = cv2.imread(input_image_path)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)  # Convert to RGB
        output_filename = os.path.splitext(os.path.basename(input_image_path))[0]

        print("Original Image\n")
        cv2_imshow(img_original)

        for i in range(1, repetitions + 1):
            N = i * amplitude
            transformation_folder = os.path.join(output_folder, transformation)
            os.makedirs(transformation_folder, exist_ok=True)
            output_subfolder = os.path.join(transformation_folder, f'n{N}')
            os.makedirs(output_subfolder, exist_ok=True)
            print(f"\nProcessing Transformations for N={N}\n")

            original_output_path = os.path.join(output_subfolder, f"{output_filename}_v{N}.jpg")
            save_image(img_original_rgb, original_output_path)  # Save original as PIL
            process_and_save_images(img_original, output_subfolder, output_filename, N, alpha_range, beta_range, gamma_range, transformation)

def show_image_transformations_help():
    help_text = """
    Image Transformation Script

    Usage:
        python main_transformer.py -i INPUT_IMAGE_PATH -o OUTPUT_FOLDER -a AMPLITUDE -r REPETITIONS -t TRANSFORMATION -c CLEAN_FOLDER

    Arguments:
        -i, --input_image_path   Path to the input image (required)
        -o, --output_folder      Path to the output folder (required)
        -a, --amplitude          Scaling factor determining the step size for the number of versions generated in each block N (required)
        -r, --repetitions        Number of blocks of transformed images to generate (required)
        -t, --transformation     Type of transformation (B, C, G, BC, BG, CG, BCG) (required)
        -c, --clean_folder       Clean the output folder before processing (optional, default: False)

    Example:
        python main_transformer.py -i /path/to/your/image.jpg -o /path/to/output/folder -a 5 -r 7 -t BCG -c True

    Description:
        This script applies a series of random transformations (contrast, brightness and gamma)
        to an input image, generating N versions of the transformed image in each block.
        The script will create subfolders within the output folder for each set of N transformed images.

    Dependencies:
        - opencv-python-headless
        - Pillow
        - numpy
        - google.colab (for mounting Google Drive and displaying images in Colab)
    """
    print(help_text)
