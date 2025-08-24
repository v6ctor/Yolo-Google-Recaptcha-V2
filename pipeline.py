from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile
from model import YOLO, Loss
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    intersection_over_union,
    plot_image,
    non_max_suppression,
)

import torch
import torch.optim as optim
import multiprocessing
import config as config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from collections import defaultdict
import random
# from torchinfo import summary

multiprocessing.set_start_method('spawn', force=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class InferenceDataset(Dataset):
    def __init__(self, img_dir, transformations=None):
        self.image_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)
                            if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transformations = transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.transformations:
            image = self.transformations(image=image)["image"]
        return image, os.path.basename(self.image_paths[idx])

class YOLODataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        img_list_file,
        anchors,
        image_size=config.IMAGE_SIZE,
        S=[13, 26, 52],
        C=3,
        transformations=None,
    ):
        with open(img_list_file, "r") as f:
            self.image_filenames = [line.strip() for line in f.readlines()]

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transformations = transformations
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_filename = self.image_filenames[index]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(
            self.label_dir, os.path.splitext(img_filename)[0] + ".txt"
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        bboxes = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h, int(class_id)])

        if self.transformations:
            augmentations = self.transformations(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            x, y, w, h, class_id = box

            box_wh = torch.tensor(box[2:4])

            box_tensor = torch.cat([torch.tensor([0, 0]), box_wh], dim=0).unsqueeze(0)

            anchors_tensor = torch.cat([torch.zeros(self.anchors.shape[0], 2), self.anchors], dim=1)

            iou_anchors = intersection_over_union(box_tensor, anchors_tensor)

            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    w_cell, h_cell = w * S, h * S
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_id)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
    
    def get_image_classes(self, index):
        img_filename = self.image_filenames[index]
        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + ".txt")

        classes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id = int(line.strip().split()[0])
                    classes.append(class_id)
        return list(set(classes))

class Pipeline:
    def __init__(self, model, loss_fn):
        self.train_dataset = None
        self.test_dataset = None
        self.train_eval_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_eval_loader = None
        self.model = model
        self.loss_fn = loss_fn

    def load_datasets(self):
   
        target_classes = list(range(3))
        samples_per_class = 500

        full_dataset = YOLODataset(
            label_dir=config.GOOGLE_TRAIN_ANNOTATIONS_DIR if len(config.LABELS_USED) == 3 else config.TRAIN_ANNOTATIONS_DIR,
            img_list_file=config.GOOGLE_TRAIN_IMG_LST_PATH if len(config.LABELS_USED) == 3 else config.TRAIN_IMG_LST_PATH,
            transformations=config.train_transformations,
            S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
            img_dir=config.GOOGLE_IMG_DIR if len(config.LABELS_USED) == 3 else config.TRAIN_IMG_DIR,
            anchors=config.ANCHORS,
        )

        selected_indices_per_class = defaultdict(list)
        class_indices = defaultdict(list)
        used_indices = set()

        for idx in range(len(full_dataset)):
            classes = full_dataset.get_image_classes(idx)
            for cls in target_classes:
                if cls in classes and idx not in used_indices:
                    class_indices[cls].append(idx)
                    used_indices.add(idx)
                    break

        final_selected_indices = set()
        for cls in target_classes:
            indices = class_indices[cls]
            random.shuffle(indices)
            count = 0
            for idx in indices:
                if idx not in final_selected_indices:
                    final_selected_indices.add(idx)
                    selected_indices_per_class[cls].append(idx)
                    count += 1
                if count >= samples_per_class:
                    break

        for cls in target_classes:
            print(f"Class {cls} — selected {len(selected_indices_per_class[cls])} images")

        self.train_dataset = torch.utils.data.Subset(full_dataset, list(final_selected_indices))

        self.test_dataset = InferenceDataset(
            transformations=config.test_transformations,
            img_dir=config.GOOGLE_TEST_IMG_DIR if len(config.LABELS_USED) == 3 else config.TEST_IMG_DIR,
        )

        self.train_eval_dataset = YOLODataset(
            label_dir=config.GOOGLE_VAL_ANNOTATIONS_DIR if len(config.LABELS_USED) == 3 else config.VAL_ANNOTATIONS_DIR,
            img_list_file=config.GOOGLE_VAL_IMG_LST_PATH if len(config.LABELS_USED) == 3 else config.VAL_IMG_LST_PATH,
            transformations=config.train_transformations,
            S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
            img_dir=config.GOOGLE_IMG_DIR if len(config.LABELS_USED) == 3 else config.VAL_IMG_DIR,
            anchors=config.ANCHORS,
        )

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

        self.train_eval_loader = DataLoader(
            dataset=self.train_eval_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

    def train(self, optimizer, scaler, scaled_anchors):
        log_file = open("training_log.csv", "a")
        log_file.write("Epoch,Train Loss,Class Accuracy,No Object Accuracy,Object Accuracy\n")

        train_losses = []
        class_accuracies = []
        no_acc_l = []
        o_acc_l = []

        for epoch in range(config.NUM_EPOCHS):
            loop = tqdm(self.train_loader, leave=True)
            batch_losses = []

            for _, (x, y) in enumerate(loop):
                x = x.to(config.DEVICE)
                y0, y1, y2 = (
                    y[0].to(config.DEVICE),
                    y[1].to(config.DEVICE),
                    y[2].to(config.DEVICE),
                )

                with torch.amp.autocast("cuda"):
                    out = self.model(x)

                    loss = (
                        self.loss_fn(out[0], y0, scaled_anchors[0])
                        + self.loss_fn(out[1], y1, scaled_anchors[1])
                        + self.loss_fn(out[2], y2, scaled_anchors[2])
                    )

                batch_losses.append(loss.item())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                mean_loss = sum(batch_losses) / len(batch_losses)
                loop.set_postfix(epoch=epoch, loss=mean_loss)

            train_losses.append(mean_loss)

            if epoch % 3 == 0 and epoch > 0:
                ca, no_acc, o_acc = check_class_accuracy(self.model, self.train_eval_loader, threshold=config.CONFIDENCE_THRESHOLD)
                class_accuracies.append(ca)
                no_acc_l.append(no_acc)
                o_acc_l.append(o_acc)

                print(f"[Epoch {epoch}] Class Acc: {ca:.4f} | Obj Acc: {o_acc:.4f} | No Obj Acc: {no_acc:.4f}")
                log_file.write(f"{epoch},{mean_loss:.4f},{ca:.4f},{no_acc:.4f},{o_acc:.4f}\n")
            else:
                log_file.write(f"{epoch},{mean_loss:.4f},,,\n")

            log_file.flush()

            if len(config.LABELS_USED) == 3:
                save_checkpoint(self.model, optimizer=optimizer, filename=config.GOOGLE_CHECKPOINT_FILE)
            else:
                save_checkpoint(self.model, optimizer=optimizer, filename=config.COCO_CHECKPOINT_FILE)

        # ======= Final mAP Evaluation After Training =======
        self.model.eval()
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.train_eval_loader,
            self.model,
            iou_threshold=config.NMS_IOU_THRESHOLD,
            anchors=config.ANCHORS,
            threshold=config.CONFIDENCE_THRESHOLD,
        )

        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESHOLD,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )

        print(f"[FINAL] mAP: {mapval.item():.4f}")
        log_file.write(f"Final mAP:,{mapval.item():.4f}\n")
        log_file.close()

        self.model.train()

        # self.save_metrics(
        #     train_losses=train_losses,
        #     map_scores=[mapval.item()],
        #     class_accuracies=class_accuracies,
        #     no_object_accuracies=no_acc_l,
        #     object_accuracies=o_acc_l
        # )

    def test(self, scaled_anchors):
        self.model.eval()

        print("\n[TEST] Running inference on unlabeled data...")

        for images, filenames in tqdm(self.test_loader):
            images = images.to(config.DEVICE)
            with torch.no_grad():
                outputs = self.model(images)

            for idx in range(len(images)):
                single_image = images[idx]
                preds = [out[idx].unsqueeze(0) for out in outputs]
                bboxes = []
                for i in range(3):
                    S = preds[i].shape[2]
                    boxes_scale_i = cells_to_bboxes(preds[i], scaled_anchors[i], S, is_preds=True)
                    bboxes += boxes_scale_i[0]

                confidences = [box[1] for box in bboxes]
                if confidences:
                    print(f"Max: {max(confidences):.4f}, Min: {min(confidences):.4f}, Mean: {sum(confidences)/len(confidences):.4f}")
                else:
                    print("No boxes found")

                top_boxes = sorted(bboxes, key=lambda x: x[1], reverse=True)[:10]
                plot_image(single_image.permute(1, 2, 0).cpu(), top_boxes, fname=filenames[idx], output_dir="debug_outputs/")
                print(f"Raw boxes (pre-NMS): {len(bboxes)}")
                nms_boxes = non_max_suppression(bboxes, config.NMS_IOU_THRESHOLD, config.CONFIDENCE_THRESHOLD, box_format="midpoint")
                print(f"NMS boxes (post-NMS): {len(nms_boxes)}")
                plot_image(single_image.permute(1, 2, 0).cpu(), nms_boxes, fname=filenames[idx], output_dir=config.DATASET + "/inference_outputs/")

        self.model.train()

   
    # def tune(self):
    #     # helper function to tune yolo hyperparameters utilizing custom genetic algorithm
    #     pass

        # Default config values for reference


DEFAULTS = {
    "lr": 1e-4, #learning rate
    "conf_thresh": 0.4,
    "nms_thresh": 0.45,
    "weight_decay": 5e-4,
    "loss_obj": 1.0,
    "loss_cls": 1.0,
    "loss_box": 1.0,
}

# Create a random individual (a set of hyperparameters) for genetic algorithm
def random_individual():
    return {
        "lr": random.uniform(5e-5, 5e-4),
        "conf_thresh": random.uniform(0.3, 0.6),
        "nms_thresh": random.uniform(0.3, 0.6),
        "weight_decay": random.uniform(1e-5, 1e-3),
        "loss_obj": random.uniform(0.8, 2.0),
        "loss_cls": random.uniform(0.8, 2.0),
        "loss_box": random.uniform(0.8, 2.0),
    }


def crossover(parent1, parent2):
    return {k: random.choice([parent1[k], parent2[k]]) for k in parent1}


def mutate(indiv, mutation_rate=0.2):
    for k in indiv:
        if random.random() < mutation_rate:
            if k == "lr":
                indiv[k] = random.uniform(5e-5, 5e-4)
            elif k in ["conf_thresh", "nms_thresh"]:
                indiv[k] = random.uniform(0.3, 0.6)
            elif k == "weight_decay":
                indiv[k] = random.uniform(1e-5, 1e-3)
            else:
                indiv[k] = random.uniform(0.8, 2.0)
    return indiv

# Train and evaluate the YOLO model w/ given hyperparameters, return mAP
def evaluate_yolo_model(**hyperparams):
    model = YOLO(conv_params=config.ARCHITECTURE, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")

    loss_fn = Loss(
        lambda_class=hyperparams["loss_cls"],
        lambda_obj=hyperparams["loss_obj"],
        lambda_box=hyperparams["loss_box"]
    )

    pipeline = Pipeline(model, loss_fn)
    pipeline.load_datasets()

    # Scale anchors to the appropriate grid size
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    pipeline.train(optimizer=optimizer, scaler=scaler, scaled_anchors=scaled_anchors)
    return pipeline.test(scaled_anchors, return_map=True)

# Genetic algorithm loop to tune hyperparameters over multiple generations
def tune(num_generations=5, population_size=8, retain=0.4, mutation_rate=0.2):
    population = [random_individual() for _ in range(population_size)]

    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        # Evaluate and sort population by performance
        scores = [(evaluate_yolo_model(**indiv), indiv) for indiv in population]
        scores.sort(reverse=True, key=lambda x: x[0])
        top_parents = [indiv for _, indiv in scores[:int(population_size * retain)]]

        # Generate next generation
        next_gen = top_parents.copy()
        while len(next_gen) < population_size:
            child = mutate(crossover(*random.sample(top_parents, 2)), mutation_rate)
            next_gen.append(child)

        population = next_gen

    # Return best individual based on highest mAP score
    best_score, best_indiv = max(scores, key=lambda x: x[0])
    print(f"\nBest mAP: {best_score:.4f} with hyperparams: {best_indiv}")
    return best_indiv


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("Tuning hyperparameters using custom genetic algorithm...")
    best_hyperparams = tune()

    logging.info(f"Best hyperparameters found: {best_hyperparams}")

    model = YOLO(conv_params=config.ARCHITECTURE, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams["lr"], weight_decay=best_hyperparams["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")

    loss_fn = Loss(
        lambda_class=hyperparams["loss_cls"],
        lambda_obj=hyperparams["loss_obj"],
        lambda_box=hyperparams["loss_box"]
    )

    pipeline = Pipeline(model, loss_fn)
    pipeline.load_datasets()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    logging.info("Starting training with best-tuned hyperparameters...")
    pipeline.train(optimizer=optimizer, scaler=scaler, scaled_anchors=scaled_anchors)

    save_checkpoint(model=model, optimizer=optimizer, filename="4_29_yolov3_model.pth.tar")


    logging.info("Starting inference on test data...")
    pipeline.test(scaled_anchors)

# def main():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler("training.log"),
#             logging.StreamHandler()
#         ]
#     )

#     logging.info("Initializing model architecture...")
#     model = YOLO(conv_params=config.ARCHITECTURE, num_classes=config.NUM_CLASSES)
#     model = model.to(config.DEVICE)

#     optimizer = optim.Adam(
#         model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
#     )

#     scaler = torch.amp.GradScaler("cuda")
#     loss_fn = Loss()

#     pipeline = Pipeline(model, loss_fn)

#     logging.info("Loading datasets...")
#     pipeline.load_datasets()

#     logging.info("Scaling anchors...")
#     scaled_anchors = (
#         torch.tensor(config.ANCHORS)
#         * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#     ).to(config.DEVICE)

#     pipeline.continue_training_model(config.GOOGLE_CHECKPOINT_FILE)

#     #logging.info("Starting training...")
#     #pipeline.train(optimizer=optimizer, scaler=scaler, scaled_anchors=scaled_anchors)

#     logging.info("Starting inference on test data...")
#     pipeline.test(scaled_anchors)

if __name__ == "__main__":
    main()
