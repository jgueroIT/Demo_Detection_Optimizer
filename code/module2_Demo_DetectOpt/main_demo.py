import argparse
import os
from demo_DetectOpt import (
    mount_drive, load_image_paths, process_image_with_yolo, show_processed_image,
    match_detections, draw_detections, print_results, save_to_excel, proccess_transformed_versions, YOLO, show_help_demo_DetectOpt)

def main():
    #drive.mount('/content/drive')
    # 1. Parse arguments from the command line
    parser = argparse.ArgumentParser(description='YOLOv8 Detection Script')
    parser.add_argument('--N', type=int, required=True, help='Number of images to process')
    parser.add_argument('--MODEL_NAME', type=str, required=True, help='YOLO model name')
    parser.add_argument('--PROJECT_NAME', type=str, required=True, help='Project name')
    parser.add_argument('--INPUT_FOLDER', type=str, required=True, help='Input folder path')
    parser.add_argument('--PROJECT_PATH', type=str, required=True, help='Project path')
    parser.add_argument('--ORIGINAL_IMAGE_PATH', type=str, required=True, help='Original image path')
    parser.add_argument('--OUTPUT_RESULT_FOLDER', type=str, required=True, help='Output result folder path')
    parser.add_argument('--IOU_THRESHOLD', type=float, default=0.50, help='IOU threshold')
    parser.add_argument('--SCORES_THRESHOLD', type=float, default=0.50, help='Scores threshold')
    parser.add_argument('--MATCHING_IOU_THRESHOLD', type=float, default=0.5, help='Matching IOU threshold')
    parser.add_argument('--CLEAR_CSV', action='store_true', help='Clear CSV before writing new data')
    parser.add_argument('--help_demo', action='store_true', help='Show help message and exit')

    args = parser.parse_args()
    
    if args.help_demo:
        show_help_demo_DetectOpt()
        exit(0)
    # 2. Assign parsed arguments to variables
    N = args.N
    MODEL_NAME = args.MODEL_NAME
    PROJECT_NAME = args.PROJECT_NAME
    INPUT_FOLDER = f"{args.INPUT_FOLDER}/n{N}"  # Append n{N} to the input folder path
    PROJECT_PATH = f"{args.PROJECT_PATH}/n{N}"  # Append n{N} to the project path
    ORIGINAL_IMAGE_PATH = args.ORIGINAL_IMAGE_PATH
    OUTPUT_RESULT_FOLDER = args.OUTPUT_RESULT_FOLDER
    IOU_THRESHOLD = args.IOU_THRESHOLD
    SCORES_THRESHOLD = args.SCORES_THRESHOLD
    MATCHING_IOU_THRESHOLD = args.MATCHING_IOU_THRESHOLD
    CLEAR_CSV = args.CLEAR_CSV 

    # 3. Create the YOLO model and get class names
    model = YOLO(MODEL_NAME)
    class_names = model.names

    # 4. Set result paths
    RESULTS_FOLDER = f"{OUTPUT_RESULT_FOLDER}/results/results_n{N}"
    SAVE_PATH_FILTERED_IMAGE = f"{RESULTS_FOLDER}/TTA_Filtered_NMS_n{N}.jpg"
    SCATTER_PLOT_PATH = f"{RESULTS_FOLDER}/Scatter_Plot_n{N}.jpg"
    EXCEL_FILE_PATH = f"{OUTPUT_RESULT_FOLDER}/results/results.csv"

    # 5. Create the results folder if it does not exist
    os.makedirs(OUTPUT_RESULT_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # 6. Load the image paths from the INPUT_FOLDER
    images_list = load_image_paths(INPUT_FOLDER)

    # 7. Process transformed versions to get detections using TTA (Test-Time Augmentation)
    TTA_Filtered_NMS_detections = proccess_transformed_versions(images_list, PROJECT_PATH, IOU_THRESHOLD, process_image_with_yolo, model, SCORES_THRESHOLD)

    # 8. Process the original image with YOLOv8 to get detections
    results_YOLOv8_Predict = process_image_with_yolo(ORIGINAL_IMAGE_PATH, RESULTS_FOLDER, f"YOLOv8_Predict_n{N}", model, SCORES_THRESHOLD)

    # 9. Show the processed image with YOLOv8 detections
    show_processed_image(os.path.join(RESULTS_FOLDER, f"YOLOv8_Predict_n{N}", os.listdir(os.path.join(RESULTS_FOLDER, f"YOLOv8_Predict_n{N}"))[0]), f"YOLOv8_Predict_n{N} Detections")

    # 10. Process the original image with YOLOv8 using augmentation to get detections
    results_YOLOv8_Predict_Aug = model.predict(ORIGINAL_IMAGE_PATH, save=True, augment=True, classes=[0, 1, 2, 3, 5, 6, 7], conf=SCORES_THRESHOLD, project=RESULTS_FOLDER, name=f"YOLOv8_Predict_Aug_n{N}")

    # 11. Show the processed image with YOLOv8 augmented detections
    show_processed_image(os.path.join(RESULTS_FOLDER, f"YOLOv8_Predict_Aug_n{N}", os.listdir(os.path.join(RESULTS_FOLDER, f"YOLOv8_Predict_Aug_n{N}"))[0]), f"YOLOv8_Predict_Aug_n{N} Detections")

    # 12. Match detections between YOLOv8, YOLOv8 with augmentation, and TTA
    matched_detections, unmatched_detections = match_detections(results_YOLOv8_Predict, results_YOLOv8_Predict_Aug, TTA_Filtered_NMS_detections, MATCHING_IOU_THRESHOLD)

    # 13. Draw detections on the original image and save the image with detections
    draw_detections(ORIGINAL_IMAGE_PATH, TTA_Filtered_NMS_detections, unmatched_detections["unmatched_TTA_Filtered_NMS"], SAVE_PATH_FILTERED_IMAGE, class_names, N)

    # 14. Print the detection results on CLI
    print_results(matched_detections, unmatched_detections, images_list, class_names, SCATTER_PLOT_PATH)

    # 15. Save the results to an Excel file
    save_to_excel(matched_detections, unmatched_detections, images_list, PROJECT_NAME, N, class_names, EXCEL_FILE_PATH, CLEAR_CSV)

# 16. Execute the main function if this script is being run as the main script
if __name__ == "__main__":
    main()
