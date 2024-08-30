# Demo_Detection_Optimizer -Traffic Object Detection Optimization with YOLOv8
This project aims to improve object detection in traffic images using YOLOv8, TTA and NMS

## Context

In recent years, Artificial Intelligence (AI) has become a transformative technology impacting various sectors globally. One area where AI has shown significant potential is object detection in images using Convolutional Neural Networks (CNNs). This project focuses on optimizing traffic object detection, such as vehicles and pedestrians, using the latest YOLOv8 model.

YOLO (You Only Look Once) is a well-known neural network architecture in object detection, with YOLOv8 being its latest iteration, offering improvements in speed and accuracy. This project leverages additional techniques, Test-Time Augmentation (TTA) and Non-Maximum Suppression (NMS), to further enhance YOLOv8's performance on traffic images.

## Objective

The goal of this project is to optimize YOLOv8's object detection predictions through TTA and NMS techniques. The project involves:

- Generating multiple transformed versions of input traffic images using variations in brightness, contrast, and gamma.
- Applying the NMS algorithm to the set of transformed images to filter redundant detections and retain the most reliable ones.
- Comparing the performance of the proposed model (**TTA_Filtered_NMS**) against the baseline YOLOv8 without TTA and NMS.
- Validating the results using real and simulated images with performance metrics.
- Assessing the impact of brightness, contrast, and gamma transformations on detection accuracy to identify the best parameters.

The project compares the performance of three models (TTA_Filtered_NMS, YOLOv8_Predict, YOLOv8_Predict_Aug) to demonstrate the effectiveness of the proposed techniques in improving detection accuracy.

## Project Structure

### Module 1 - Image Transformations

**Purpose:** Apply TTA by generating multiple versions of a traffic image with variations in brightness, contrast, and gamma.

**Scripts:**

- `image_transformer.py`: Functions to apply transformations and manage file I/O.
- `main_transformer.py`: Executes the script, processes input arguments, and manages the workflow.

**Input Arguments:**

- `INPUT_IMAGE_PATH`: Path to the input image ('.jpg' or '.png').
- `OUTPUT_FOLDER`: Directory to save transformed images.
- `AMPLITUDE`: Scaling factor for the number of generated versions per block.
- `REPETITIONS`: Number of times to generate a set of transformed images.
- `TRANSFORMATION`: Type of transformation to apply (options: B, C, G, BC, BG, CG, BCG).
- `CLEAN_FOLDER`: Optional flag to clear the output folder before processing.

### Module 2 - Demo_DetectOpt

**Purpose:** Use YOLOv8's predict method to detect objects in traffic images and apply TTA and NMS techniques to improve predictions.

**Scripts:**

- `demo_DetectOpt.py`: Functions for object detection and file management.
- `main_demo.py`: Main script for processing input arguments.

**Input Arguments:**

- `N`: Number of images to process.
- `MODEL_NAME`: Name of the pre-trained model (e.g., "yolov8n.pt").
- `PROJECT_NAME`: Name of the simulation project.
- `INPUT_FOLDER`: Directory containing transformed images.
- `PROJECT_PATH`: Directory for storing intermediate results.
- `ORIGINAL_IMAGE_PATH`: Path to the original image used for reference.
- `OUTPUT_RESULT_FOLDER`: Directory for saving results and visualizations.
- `IOU_THRESHOLD`: IOU threshold value for the NMS algorithm.
- `SCORES_THRESHOLD`: Minimum confidence score for valid detections.
- `MATCHING_IOU_THRESHOLD`: IOU threshold for matching detections across models.
- `CLEAR_CSV`: Optional flag to clear the output CSV file before writing new data.

## Libraries Used

- `os`, `pathlib`: System operations and file handling.
- `OpenCV`, `PIL`: Image processing and transformations.
- `numpy`: Numerical computations.
- `ultralytics`: Contains the YOLOv8 model for object detection.
- `matplotlib.pyplot`: Visualization of processed images and detection results.
- `torch`: PyTorch for handling tensors and deep learning operations.
- `pandas`: Data manipulation and analysis.
- `re`: Text manipulation based on patterns.
- `argparse`: Command-line argument parsing.
- `google.colab`: Integration with Google Colab for cloud-based execution.

## Installation and Usage

To use this project, follow these steps:

### Setup Environment

Install necessary dependencies:

```bash
pip install opencv-python-headless Pillow numpy ultralytics matplotlib torch pandas

### Run Module 1. Execute `main_transformer.py` to apply image transformations:

```bash
python main_transformer.py --INPUT_IMAGE_PATH <path_to_input_image> --OUTPUT_FOLDER <path_to_output_folder> --AMPLITUDE <num> --REPETITIONS <num> --TRANSFORMATION <B, C, G, BC, BG, CG or BCG> -c

### Run Module 2. Execute main_demo.py to perform object detection:

```bash
python main_demo.py --N <number_of_images> --MODEL_NAME yolov8n.pt --PROJECT_NAME <project_name> --INPUT_FOLDER <path_to_input_folder> --PROJECT_PATH <path_to_project_folder> --ORIGINAL_IMAGE_PATH <path_to_original_image> --OUTPUT_RESULT_FOLDER <path_to_results_folder> --IOU_THRESHOLD <iou_value> --SCORES_THRESHOLD <score_value> --MATCHING_IOU_THRESHOLD <matching_iou_value> --CLEAR_CSV

