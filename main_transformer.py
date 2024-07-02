import argparse
from google.colab import drive
import sys
from google.colab.patches import cv2_imshow
from image_transformer import transform_image, show_image_transformations_help

def main():
    parser = argparse.ArgumentParser(description="Image Transformation Script")
    
    parser.add_argument('--INPUT_IMAGE_PATH', '-i', type=str, required=True, help='Path to the input image')
    parser.add_argument('--OUTPUT_FOLDER', '-o', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--AMPLITUDE', '-a', type=int, required=True, help='Scaling factor determining the step size for the number of versions generated in each block N')
    parser.add_argument('--REPETITIONS', '-r', type=int, required=True, help='Number of blocks of transformed images to generate')
    parser.add_argument('--TRANSFORMATION', '-t', type=str, required=True, choices=['B', 'C', 'G', 'BC', 'BG', 'CG', 'BCG'], help='Type of transformation to apply (B, C, G, BC, BG, CG, BCG)')
    parser.add_argument('--CLEAN_FOLDER', '-c', action='store_true', help='Clean the output folder before processing')
    parser.add_argument('--HELP_TRANSFORMATIONS', '-ht', action='store_true', help='Show help for image transformations')

    try:
        args = parser.parse_args()

        if args.HELP_TRANSFORMATIONS:
            show_image_transformations_help()
        else:
            #drive.mount('/content/drive')
            transform_image(args.INPUT_IMAGE_PATH, args.OUTPUT_FOLDER, args.AMPLITUDE, args.REPETITIONS, args.TRANSFORMATION, args.CLEAN_FOLDER)

    except SystemExit as e:
        if e.code != 0:
            parser.print_help()
        sys.exit(e.code)

if __name__ == "__main__":
    main()