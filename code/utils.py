import config
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import logging
import json
import csv
import pandas as pd

logging.basicConfig(level=logging.INFO)

def coco_to_csv(coco_json_path, images_dir, output_csv):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    image_id_map = {
        img['id']: {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        } for img in coco['images']
    }

    rows = []
    print("Processing and normalizing annotations...")
    for ann in tqdm(coco['annotations']):
        image_id = ann['image_id']
        image_info = image_id_map[image_id]
        filename = image_info['file_name']
        img_w = image_info['width']
        img_h = image_info['height']

        bbox = ann['bbox']  # [x_min, y_min, width, height]
        category_id = ann['category_id']

        x_min, y_min, width, height = bbox
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        x_center /= img_w
        y_center /= img_h
        width /= img_w
        height /= img_h

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            continue

        rows.append({
            "image": os.path.join(images_dir, filename),
            "class_id": category_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"✅ Normalized COCO CSV saved to {output_csv}")

def coco_to_yolo(coco_json_path, output_dir, img_dir):
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}
    category_name_to_id = {name: i for i, (id, name) in enumerate(category_id_to_name.items())}

    for image in coco_data['images']:
        img_id = image['id']
        img_filename = image['file_name']
        img_path = os.path.join(img_dir, img_filename)

        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, 'w') as txt_file:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == img_id:
                    category_id = annotation['category_id']
                    category_name = category_id_to_name[category_id]
                    yolo_class = category_name_to_id[category_name]

                    bbox = annotation['bbox']
                    x, y, width, height = bbox

                    x_center = x + width / 2
                    y_center = y + height / 2
                    x_center /= image['width']
                    y_center /= image['height']
                    width /= image['width']
                    height /= image['height']

                    txt_file.write(f"{yolo_class} {x_center} {y_center} {width} {height}\n")

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        chosen_tensor = torch.tensor(chosen_box[2:])

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
            intersection_over_union(chosen_tensor, torch.tensor(box[2:]), box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [gt for gt in true_boxes if gt[1] == c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key in amount_bboxes:
            amount_bboxes[key] = torch.zeros(amount_bboxes[key])

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        if len(ground_truths) == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            gt_img = [gt for gt in ground_truths if gt[0] == detection[0]]
            best_iou, best_gt_idx = 0, -1

            for idx, gt in enumerate(gt_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold and amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (len(ground_truths) + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0

def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0

    for x, labels in tqdm(loader):
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor(anchors[i]).to(device) * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S, is_preds=True)
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold, threshold, box_format)
            all_pred_boxes += [[train_idx] + box for box in nms_boxes]
            all_true_boxes += [[train_idx] + box for box in true_bboxes[idx] if box[1] > threshold]
            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    B, A = predictions.shape[0], len(anchors)
    box_preds = predictions[..., 1:5]

    if is_preds:
        anchors = anchors.reshape(1, A, 1, 1, 2)
        box_preds[..., 0:2] = torch.sigmoid(box_preds[..., 0:2])
        box_preds[..., 2:] = torch.exp(box_preds[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = torch.arange(S).repeat(B, A, S, 1).unsqueeze(-1).to(predictions.device)
    x = (box_preds[..., 0:1] + cell_indices) / S
    y = (box_preds[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / S
    wh = box_preds[..., 2:4] / S

    converted = torch.cat((best_class, scores, x, y, wh), dim=-1).reshape(B, A * S * S, 6)
    return converted.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    correct_class = correct_noobj = correct_obj = 0
    tot_class_preds = tot_noobj = tot_obj = 0

    for x, y in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            correct_class += (torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]).sum()
            tot_class_preds += obj.sum()

            preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += (preds[obj] == y[i][..., 0][obj]).sum()
            tot_obj += obj.sum()
            correct_noobj += (preds[noobj] == y[i][..., 0][noobj]).sum()
            tot_noobj += noobj.sum()

    print(f"Class accuracy: {(correct_class / (tot_class_preds + 1e-16)) * 100:.2f}%")
    print(f"No obj accuracy: {(correct_noobj / (tot_noobj + 1e-16)) * 100:.2f}%")
    print(f"Obj accuracy: {(correct_obj / (tot_obj + 1e-16)) * 100:.2f}%")
    model.train()

    return (correct_class / (tot_class_preds + 1e-16)) * 100,(correct_noobj / (tot_noobj + 1e-16)) * 100,(correct_obj / (tot_obj + 1e-16)) * 100

def get_mean_std(loader):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += data.mean(dim=[0, 2, 3])
        channels_sqrd_sum += (data ** 2).mean(dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2).sqrt()
    return mean, std

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    logging.info("Saving checkpoint...")
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename)

def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Loaded model state dict")

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> Loaded optimizer state dict")

        if lr is not None:
            print(f"=> Overriding LR to {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors, class_labels=None):
    model.eval()

    try:
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
    except Exception as e:
        raise ValueError(f"Failed to get data from loader: {e}")

    x = x.to("cuda")

    with torch.no_grad():
        out = model(x)

    bboxes = [[] for _ in range(x.shape[0])]
    for i in range(3):
        S = out[i].shape[2]
        boxes_scale_i = cells_to_bboxes(out[i], anchors[i], S, is_preds=True)
        for idx, box in enumerate(boxes_scale_i):
            bboxes[idx] += box

    model.train()

    for i in range(x.shape[0]):
        nms_boxes = non_max_suppression(bboxes[i], iou_thresh, thresh, box_format="midpoint")
        plot_image(x[i].permute(1, 2, 0).cpu(), nms_boxes, class_labels=class_labels)

def plot_image(image, boxes, fname=None, output_dir="inference_outputs"):
    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS_USED
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

    fig, ax = plt.subplots(1)
    ax.imshow(image.numpy())

    for box in boxes:
        class_pred = int(box[0])
        conf = float(box[1])  # Confidence score
        x, y, w, h = box[2:]
        ul_x, ul_y = x - w / 2, y - h / 2

        rect = patches.Rectangle(
            (ul_x * image.shape[1], ul_y * image.shape[0]),
            w * image.shape[1],
            h * image.shape[0],
            linewidth=2,
            edgecolor=colors[class_pred],
            facecolor="none"
        )
        ax.add_patch(rect)

        # Label with class name and confidence
        label = f"{class_labels[class_pred]}: {conf:.2f}"
        ax.text(
            ul_x * image.shape[1],
            ul_y * image.shape[0],
            s=label,
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred], "pad": 0},
        )

    os.makedirs(output_dir, exist_ok=True)
    if fname is None:
        fname = "output.png"
    save_path = os.path.join(output_dir, fname)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# import os
# import shutil

# # Base path where the folders are
# base_path = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/labels"
# output_folder = os.path.join(base_path, "google_train2025")

# # Create output directory
# os.makedirs(output_folder, exist_ok=True)

# # Get all subdirectories (excluding google_train2025 itself)
# subfolders = [f for f in os.listdir(base_path) 
#               if os.path.isdir(os.path.join(base_path, f)) and f != "google_train2025"]

# # Take only the first 3 folders
# selected_folders = subfolders[:3]

# print(f"Combining .txt files from: {selected_folders}")

# # Move or copy files
# for folder in selected_folders:
#     folder_path = os.path.join(base_path, folder)
#     for file in os.listdir(folder_path):
#         if file.endswith(".txt"):
#             src = os.path.join(folder_path, file)
#             dst = os.path.join(output_folder, file)
#             # If duplicate file names exist, you can rename them here to avoid overwrite
#             if os.path.exists(dst):
#                 base, ext = os.path.splitext(file)
#                 count = 1
#                 while os.path.exists(dst):
#                     dst = os.path.join(output_folder, f"{base} {count}{ext}")
#                     count += 1
#             shutil.copy(src, dst)  # Use .move(src, dst) if you want to move instead
#             print(f"Copied: {src} → {dst}")

# print("Step 1 complete: All .txt files are in google_train2025.")

# import os
# import shutil

# # Base directory
# base_dir = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/images"
# output_dir = os.path.join(base_dir, "google_train_images_2025")

# # Ensure destination exists
# os.makedirs(output_dir, exist_ok=True)

# # Folders to include
# folders_to_copy = ["Chimney", "Crosswalk", "Stair"]

# # Image extensions to include
# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# # Go through each folder and copy images
# for folder in folders_to_copy:
#     folder_path = os.path.join(base_dir, folder)
#     for file in os.listdir(folder_path):
#         if os.path.splitext(file)[1].lower() in image_extensions:
#             src = os.path.join(folder_path, file)
#             dst = os.path.join(output_dir, file)

#             # Avoid overwriting if duplicate filenames exist
#             if os.path.exists(dst):
#                 base, ext = os.path.splitext(file)
#                 count = 1
#                 while os.path.exists(dst):
#                     dst = os.path.join(output_dir, f"{base}_{count}{ext}")
#                     count += 1

#             shutil.copy(src, dst)
#             print(f"Copied: {src} → {dst}")

# print("✅ Image copy complete.")

# import os
# import random
# from collections import defaultdict
# import shutil

# # Base input and output paths
# base_input = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/labels/google_train2025"
# output_base = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/split_labels"
# splits = {'train': 0.6, 'val': 0.1, 'test': 0.3}

# # Create output folders
# for split in splits:
#     os.makedirs(os.path.join(output_base, split), exist_ok=True)

# # Collect all .txt files and categorize by class (from parent folder)
# class_to_files = defaultdict(list)

# for root, dirs, files in os.walk(base_input):
#     for file in files:
#         if file.endswith(".txt"):
#             class_name = os.path.basename(root)  # Get folder name as class
#             full_path = os.path.join(root, file)
#             class_to_files[class_name].append(full_path)

# # Prepare split dictionaries
# split_files = {'train': [], 'val': [], 'test': []}

# # Process each class
# for class_name, files in class_to_files.items():
#     random.shuffle(files)
#     total = len(files)
#     n_train = int(splits['train'] * total)
#     n_val = int(splits['val'] * total)

#     split_files['train'].extend(files[:n_train])
#     split_files['val'].extend(files[n_train:n_train + n_val])
#     split_files['test'].extend(files[n_train + n_val:])

# # Copy to split folders
# for split, files in split_files.items():
#     for file_path in files:
#         dest_path = os.path.join(output_base, split, os.path.basename(file_path))
#         shutil.copy(file_path, dest_path)

# print("Finished splitting .txt files into train/val/test with class balance.")

# import os
# import shutil

# label_dir = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/split_labels/val"
# image_source_dir = "/home/ndandre1/scratch/coco2017/Google_Recaptcha_V2_Images_Dataset/images/google_train_images_2025"
# output_dir = "/home/ndandre1/scratch/coco2017/yolov5_dataset/images/val"

# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
# os.makedirs(output_dir, exist_ok=True)

# for label_file in os.listdir(label_dir):
#     if label_file.endswith(".txt"):
#         base_name = os.path.splitext(label_file)[0]  # e.g., "Chimney (1)"

#         found = False
#         for ext in image_extensions:
#             image_file = f"{base_name}{ext}"
#             src_path = os.path.join(image_source_dir, image_file)
#             if os.path.exists(src_path):
#                 dst_path = os.path.join(output_dir, image_file)
#                 shutil.copy(src_path, dst_path)
#                 print(f"Copied: {src_path} → {dst_path}")
#                 found = True
#                 break

#         if not found:
#             print(f"⚠️ Image not found for label: {label_file}")

# print("✅ All matching images copied to train_img_dir.")

def put_seed(seed):
    logging.info(f"Seeding everything with seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g