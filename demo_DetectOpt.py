import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import pandas as pd
import re

def mount_drive():
    """Mount the Google Colab drive."""
    from google.colab import drive
    drive.mount('/content/drive')

def load_image_paths(input_folder):
    """Return the paths of all images in the specified folder, sorted by version number."""
    images_list = [
        os.path.join(input_folder, image_name)
        for image_name in os.listdir(input_folder)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))
    ]
    images_list.sort(key=lambda filepath: int(re.search(r'_v(\d+)', filepath).group(1)) if re.search(r'_v(\d+)', filepath) else 0)
    return images_list

def process_image_with_yolo(img, project_path, folder_name, model, conf):
    """Process an image with YOLO and return its results."""
    return model.predict(img, save=True, classes=[0, 1, 2, 3, 5, 6, 7], conf=conf, project=project_path, name=folder_name)

def show_processed_image(image_path, title):
    """Show the image at the specified path with a given title."""
    try:
        from google.colab.patches import cv2_imshow  # Importar cv2_imshow de google.colab

        img = cv2.imread(image_path)
        cv2_imshow(img)  # Usar cv2_imshow en lugar de plt.imshow para Google Colab
        print(title)  # Mostrar el título en la salida
    except Exception as e:
        print(f"Error showing the image: {e}")

def get_detections(results, img_version_idx=None):
    """Extract and return the detections from the results."""
    return [
        (xyxy, conf.item(), cls.item(), img_version_idx)
        for xyxy, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)
    ]

def nms(detections, iou_threshold):
    """Perform Non-maximum Suppression on the detections."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    new_detections = []
    while detections:
        current_detection = detections.pop(0)
        new_detections.append(current_detection)
        detections = [det for det in detections if get_iou(current_detection[0], det[0]) <= iou_threshold]
    return new_detections

def get_iou(bb1, bb2):
    """Calculate and return the Intersection over Union between two bounding boxes."""
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return intersection_area / float(bb1_area + bb2_area - intersection_area)


def draw_detections(image_path, detections, unmatched_detections, save_path, class_names, N):
    """Draw bounding boxes and labels on the image."""
    img = cv2.imread(image_path)  # Leer la imagen dentro de la función
    title =  f"TTA_Filtered_NMS Detections_n{N}"
    label_positions = []
    for xyxy, conf, cls, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)

        color = (0, 255, 0)  # Green bbox by default
        if unmatched_detections and any(np.array_equal(xyxy.cpu().numpy(), det[0].cpu().numpy()) for det in unmatched_detections):
            color = (255, 0, 0)  # Red color for unmatched detections

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{class_names[int(cls)]} {conf:.2f}" if class_names else f"{conf:.2f}"
        font_scale = 1.2
        font_thickness = 3
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_width, text_height = text_size[0], text_size[1]
        positions = [
            (x1 + 2, y1 - text_height - 2),
            (x1 + 2, y2 + text_height + 2),
            (x2 - text_width - 2, y1 - text_height - 2),
            (x2 - text_width - 2, y2 + text_height + 2),
        ]
        label_x1, label_y1 = next((pos for pos in positions if not any(
            (pos[0] < p_x2 and pos[0] + text_width > p_x1 and
             pos[1] - text_height < p_y2 and pos[1] > p_y1)
            for p_x1, p_y1, p_x2, p_y2 in label_positions)), (None, None))
        if label_x1 is None or label_y1 is None:
            label_x1, label_y1 = img.shape[1] - text_width - 10, text_height + 10
        label_positions.append((label_x1, label_y1 - text_height, label_x1 + text_width, label_y1))
        padding = 4
        cv2.rectangle(img, (label_x1 - padding, label_y1 - text_height - padding),
                      (label_x1 + text_width + padding, label_y1 + padding), color, -1)
        cv2.putText(img, label, (label_x1, label_y1 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.imwrite(save_path, img) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show_processed_image(save_path, title)  

    return img

def match_detections(results_YOLOv8_Predict, results_YOLOv8_Predict_Aug, detections_c, iou_threshold):
    """Match detections from three sets based on IOU threshold."""
    detections_a = get_detections(results_YOLOv8_Predict)
    detections_b = get_detections(results_YOLOv8_Predict_Aug)

    def get_best_match(det, dets):
        best_match = None
        best_iou = iou_threshold
        for d in dets:
            iou = get_iou(det[0], d[0])
            if iou > best_iou:
                best_iou = iou
                best_match = d
        return best_match

    matches, unmatched_a, unmatched_b, unmatched_c = [], list(detections_a), list(detections_b), list(detections_c)
    for det_a in detections_a:
        match_b = get_best_match(det_a, detections_b)
        match_c = get_best_match(det_a, detections_c)
        if match_b and match_c:
            matches.append((det_a, match_b, match_c))
            unmatched_b = [d for d in unmatched_b if not torch.equal(d[0], match_b[0])]
            unmatched_c = [d for d in unmatched_c if not torch.equal(d[0], match_c[0])]

    for det_b in unmatched_b[:]:
        match_c = get_best_match(det_b, unmatched_c)
        if match_c:
            matches.append((None, det_b, match_c))
            unmatched_c = [d for d in unmatched_c if not torch.equal(d[0], match_c[0])]

    unmatched_a = [det for det in unmatched_a if not any(torch.equal(det[0], match[0][0]) for match in matches if match[0])]
    unmatched_b = [det for det in unmatched_b if not any(torch.equal(det[0], match[1][0]) for match in matches if match[1])]
    unmatched_c = [det for det in unmatched_c if not any(torch.equal(det[0], match[2][0]) for match in matches if match[2])]

    unmatched_detections = {
        "unmatched_YOLOv8_Predict": unmatched_a,
        "unmatched_YOLOv8_Predict_Aug": unmatched_b,
        "unmatched_TTA_Filtered_NMS": unmatched_c,
    }

    return matches, unmatched_detections

def print_results(matched_detections, unmatched_detections, images_list, class_names, SCATTER_PLOT_PATH):
    """Display the results of the detections and plot the scatter plot."""
    def format_detection(det):
        score = f"{det[1]:.4f}" if det else "Not detected"
        cls_idx = det[2] if det else None
        cls_name = f"({class_names[cls_idx]})" if cls_idx is not None else ""
        return f"{score} {cls_name}"

    for match in matched_detections:
        YOLOv8_Predict_score = f"{match[0][1]:.4f}" if match[0] is not None else "Not detected"
        YOLOv8_Predict_class_index = match[0][2] if match[0] is not None else None
        YOLOv8_Predict_class = f"({class_names[YOLOv8_Predict_class_index]})" if YOLOv8_Predict_class_index is not None else ""

        YOLOv8_Predict_Aug_score = f"{match[1][1]:.4f}" if match[1] is not None else "Not detected"
        YOLOv8_Predict_Aug_class_index = match[1][2] if match[1] is not None else None
        YOLOv8_Predict_Aug_class = f"({class_names[YOLOv8_Predict_Aug_class_index]})" if YOLOv8_Predict_Aug_class_index is not None else ""

        TTA_Filtered_NMS_score = f"{match[2][1]:.4f}" if match[2] is not None else "Not detected"
        TTA_Filtered_NMS_class_index = match[2][2] if match[2] is not None else None
        TTA_Filtered_NMS_class = f"({class_names[TTA_Filtered_NMS_class_index]})" if TTA_Filtered_NMS_class_index is not None else ""
        TTA_Filtered_NMS_idx = match[2][3] if match[2] is not None else None

        if TTA_Filtered_NMS_idx is not None:
            final_image_name = os.path.basename(images_list[int(TTA_Filtered_NMS_idx)])
            print(f"YOLOv8_Predict: {YOLOv8_Predict_score} {YOLOv8_Predict_class}, YOLOv8_Predict_Aug: {YOLOv8_Predict_Aug_score} {YOLOv8_Predict_Aug_class}, TTA_Filtered_NMS: {TTA_Filtered_NMS_score} {TTA_Filtered_NMS_class} in version {final_image_name}")
        else:
            print(f"YOLOv8_Predict: {YOLOv8_Predict_score} {YOLOv8_Predict_class}, YOLOv8_Predict_Aug: {YOLOv8_Predict_Aug_score} {YOLOv8_Predict_Aug_class}, TTA_Filtered_NMS: {TTA_Filtered_NMS_score} {TTA_Filtered_NMS_class}")

    def print_unique_detections(unmatched, label, images_list):
        print(f"\nUnique detections in {label}:")
        for um in unmatched:
            um_score = f"{um[1]:.4f}"
            um_class_index = um[2]
            um_class = f"({class_names[um_class_index]})" if um_class_index is not None else ""
            um_img_version_idx = um[3] if len(um) > 3 else None
            if um_img_version_idx is not None:
                um_image_name = os.path.basename(images_list[int(um_img_version_idx)])
                print(f"{label}: {um_score} {um_class} in image {um_image_name}")
            else:
                print(f"{label}: {um_score} {um_class}")

    print_unique_detections(unmatched_detections["unmatched_YOLOv8_Predict"], "YOLOv8_Predict", images_list)
    print_unique_detections(unmatched_detections["unmatched_YOLOv8_Predict_Aug"], "YOLOv8_Predict_Aug", images_list)
    print_unique_detections(unmatched_detections["unmatched_TTA_Filtered_NMS"], "TTA_Filtered_NMS", images_list)

    plot_scatter(matched_detections, unmatched_detections, SCATTER_PLOT_PATH)

def plot_scatter(matched_detections, unmatched_detections, SCATTER_PLOT_PATH):
    """Plot a scatter plot of the detection confidences."""
    fig, ax = plt.subplots()
    thickness = 7

    for index, match in enumerate(matched_detections):
        if match[0] and match[1] and match[2] and match[0][1] == match[1][1] == match[2][1]:
            ax.scatter(index, match[0][1], color='black', s=thickness, label='All Models' if index == 0 else "")
        else:
            if match[0]:
                ax.scatter(index, match[0][1], color='blue', s=thickness, label='YOLOv8_Predict' if index == 0 else "")
            if match[1]:
                ax.scatter(index, match[1][1], color='red', s=thickness, label='YOLOv8_Predict_Aug' if index == 0 else "")
            if match[2]:
                ax.scatter(index, match[2][1], color='green', s=thickness, label='TTA_Filtered_NMS' if index == 0 else "")

    unique_index = len(matched_detections)
    for um in unmatched_detections["unmatched_YOLOv8_Predict"]:
        ax.scatter(unique_index, um[1], color='blue', s=thickness)
        unique_index += 1
    for um in unmatched_detections["unmatched_YOLOv8_Predict_Aug"]:
        ax.scatter(unique_index, um[1], color='red', s=thickness)
        unique_index += 1
    for um in unmatched_detections["unmatched_TTA_Filtered_NMS"]:
        ax.scatter(unique_index, um[1], color='green', s=thickness)
        unique_index += 1

    ax.legend()
    ax.set_title('Objects detected - Scatter Plot')
    ax.set_xlabel('Detection Index')
    ax.set_ylabel('Confidence')
    ax.set_ylim([0, 1])

    plt.savefig(SCATTER_PLOT_PATH)
    plt.show()

def calculate_mean_confidences(matched_detections):
    """Calculate mean confidence scores for matched detections across the three models."""
    yolo_predict_scores = []
    yolo_predict_aug_scores = []
    tta_filtered_nms_scores = []

    for match in matched_detections:
        if match[0]:  # YOLOv8_Predict detection exists
            yolo_predict_scores.append(match[0][1])
            yolo_predict_aug_scores.append(match[1][1] if match[1] else 0)
            tta_filtered_nms_scores.append(match[2][1] if match[2] else 0)

    mean_yolo_predict = np.mean(yolo_predict_scores) if yolo_predict_scores else 0
    mean_yolo_predict_aug = np.mean(yolo_predict_aug_scores) if yolo_predict_aug_scores else 0
    mean_tta_filtered_nms = np.mean(tta_filtered_nms_scores) if tta_filtered_nms_scores else 0

    print(f"Mean Confidence YOLOv8_Predict: {mean_yolo_predict:.4f}")
    print(f"Mean Confidence YOLOv8_Predict_Aug: {mean_yolo_predict_aug:.4f}")
    print(f"Mean Confidence TTA_Filtered_NMS: {mean_tta_filtered_nms:.4f}")

    return mean_yolo_predict, mean_yolo_predict_aug, mean_tta_filtered_nms

def calculate_new_detections(unmatched_detections):
    """Calculate and print the number of new detections in YOLOv8_Predict_Aug and TTA_Filtered_NMS."""
    new_detections_aug = len(unmatched_detections["unmatched_YOLOv8_Predict_Aug"])
    new_detections_tta = len(unmatched_detections["unmatched_TTA_Filtered_NMS"])
    print(f"New detections in YOLOv8_Predict_Aug: {new_detections_aug}")
    print(f"New detections in TTA_Filtered_NMS: {new_detections_tta}")

    return new_detections_aug, new_detections_tta

def save_to_excel(matched_detections, unmatched_detections, images_list, project_name, N, class_names, CSV_FILE_PATH, clear_csv=False):
    """Save the results to an Excel file."""
    data = []
    idx = 0  

    mean_confidences = calculate_mean_confidences(matched_detections)
    new_detections = calculate_new_detections(unmatched_detections)

    # Add matched detections
    for match in matched_detections:
        yolo_predict = match[0][1] if match[0] else 0
        yolo_predict_aug = match[1][1] if match[1] else 0
        tta_filtered_nms = match[2][1] if match[2] else 0
        object_class = class_names[match[0][2]] if match[0] else "Unknown"
        image_version_name = os.path.basename(images_list[match[2][3]]) if match[2] else "N/A"

        data.append([idx, object_class, yolo_predict, yolo_predict_aug, tta_filtered_nms, image_version_name])
        idx += 1

    # Add unique detections from YOLOv8_Predict
    for det in unmatched_detections["unmatched_YOLOv8_Predict"]:
        data.append([idx, class_names[det[2]], det[1], "", "", os.path.basename(images_list[det[3]]) if det[3] is not None else "N/A"])
        idx += 1

    # Add unique detections from YOLOv8_Predict_Aug
    for det in unmatched_detections["unmatched_YOLOv8_Predict_Aug"]:
        data.append([idx, class_names[det[2]], "", det[1], "", os.path.basename(images_list[det[3]]) if det[3] is not None else "N/A"])
        idx += 1

    # Add unique detections from TTA_Filtered_NMS
    for det in unmatched_detections["unmatched_TTA_Filtered_NMS"]:
        data.append([idx, class_names[det[2]], "", "", det[1], os.path.basename(images_list[det[3]]) if det[3] is not None else "N/A"])
        idx += 1

    # Add mean confidences row
    data.append(["Mean Confidence", "", mean_confidences[0], mean_confidences[1], mean_confidences[2], ""])

    # Add new detections row
    data.append(["New Detections", "", "", new_detections[0], new_detections[1], ""])

    # Calculate the number of detections for each model
    num_detections_yolo_predict = len(matched_detections) + len(unmatched_detections["unmatched_YOLOv8_Predict"])
    num_detections_yolo_predict_aug = len(matched_detections) + len(unmatched_detections["unmatched_YOLOv8_Predict_Aug"])
    num_detections_tta_filtered_nms = len(matched_detections) + len(unmatched_detections["unmatched_TTA_Filtered_NMS"])

    # Add number of detections row
    data.append(["Number of Detections", "", num_detections_yolo_predict, num_detections_yolo_predict_aug, num_detections_tta_filtered_nms, ""])

    df = pd.DataFrame(data, columns=["Object_Index", "Object_class", "YOLOv8_Predict", "YOLOv8_Predict_Aug", "TTA_Filtered_NMS", "image_version_name"])

    if clear_csv:
        # Clear the file if clear_csv is True
        with open(CSV_FILE_PATH, 'w') as f:
            f.truncate(0)

    # Save the DataFrame to a CSV file
    with open(CSV_FILE_PATH, 'a') as f:
        if not clear_csv and os.path.getsize(CSV_FILE_PATH) > 0:
            f.write("\n\n")  # Add 2 empty lines if appending to the file
        f.write(f"Project Name: {project_name}, N: {N}\n")
        df.to_csv(f, index=False)

    print(f"results.csv successfully generated")

    
def proccess_transformed_versions(images_list, PROJECT_PATH, IOU_THRESHOLD, process_image_with_yolo, model, SCORES_THRESHOLD):
    all_detections = []
    for idx, img_path in enumerate(images_list):
        if img_path:
            results = process_image_with_yolo(img_path, PROJECT_PATH, f"predict_{idx}", model, SCORES_THRESHOLD)
            show_processed_image(os.path.join(PROJECT_PATH, f"predict_{idx}", os.listdir(os.path.join(PROJECT_PATH, f"predict_{idx}"))[0]), f"Image {idx}")
            all_detections.extend(get_detections(results, idx))
        else:
            print(f"Could not load image {img_path}")
    return nms(all_detections, IOU_THRESHOLD)
    
def show_help_demo_DetectOpt():
    help_text = """
    Demo_DetectOpt Module - Help

    This module uses the YOLOv8 model to perform object detection in images and applies enhancement techniques such as Non-Maximum Suppression (NMS) and Test-Time Augmentation (TTA).

    Input Arguments:
    --N: Number of transformed images to process.
    --MODEL_NAME: Name of the YOLO model.
    --PROJECT_NAME: Name of the project.
    --INPUT_FOLDER: Path to the input folder containing the transformed images.
    --PROJECT_PATH: Project path.
    --ORIGINAL_IMAGE_PATH: Path to the original image.
    --OUTPUT_RESULT_FOLDER: Path to the results folder.
    --IOU_THRESHOLD: IOU (Intersection over Union) threshold value for non-maximum suppression.
    --SCORES_THRESHOLD: Confidence threshold value for detection.
    --MATCHING_IOU_THRESHOLD: IOU threshold for matching detections.
    --CLEAR_CSV: Clear the CSV file before writing new data (True/False).

    Usage Example:
    !python main_demo.py --N 5 --MODEL_NAME yolov8 --PROJECT_NAME MyProject --INPUT_FOLDER /path/to/input_folder --PROJECT_PATH /path/to/project --ORIGINAL_IMAGE_PATH /path/to/original_image.jpg --OUTPUT_RESULT_FOLDER /path/to/output_folder --IOU_THRESHOLD 0.5 --SCORES_THRESHOLD 0.5 --MATCHING_IOU_THRESHOLD 0.5 --clear_csV
    """
    print(help_text)