import albumentations as A
import cv2
import torch
import os

from albumentations.pytorch import ToTensorV2

DATASET = "/home/ndandre1/scratch/coco2017"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = os.cpu_count() - 1
BATCH_SIZE = 8
IMAGE_SIZE = 128 #416
NUM_CLASSES = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0
NUM_EPOCHS = 1000

CONFIDENCE_THRESHOLD = 0.3
MAP_IOU_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.3

S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
COCO_CHECKPOINT_FILE = DATASET + "/models/4_29_12yolov3_model.pth.tar"
GOOGLE_CHECKPOINT_FILE = DATASET + "/models/4_29_2yolov3zz_google_model.pth.tar"
TRAIN_IMG_DIR = DATASET + "/train2017/"
TEST_IMG_DIR = DATASET + "/test2017/"
VAL_IMG_DIR = DATASET + "/val2017"
LABELS_DIR = DATASET + "/labels/"
TRAIN_IMG_LST_PATH = DATASET + "/train_image_list.txt"
VAL_IMG_LST_PATH = DATASET + "/val_image_list.txt"


TRAIN_ANNOTATIONS_DIR = DATASET + "/annotations/train2017_annotations.csv"
VAL_ANNOTATIONS_DIR = DATASET + "/annotations/val2017_annotations.csv"

GOOGLE_IMG_DIR = DATASET + "/Google_Recaptcha_V2_Images_Dataset/images/google_train_images_2025/"
GOOGLE_LABELS_DIR = DATASET + "/Google_Recaptcha_V2_Images_Dataset/labels/google_train2025/"
GOOGLE_TRAIN_IMG_LST_PATH = DATASET + "/Google_Recaptcha_V2_Images_Dataset/images/train_image_list.txt"
GOOGLE_VAL_IMG_LST_PATH = DATASET + "/Google_Recaptcha_V2_Images_Dataset/images/val_image_list.txt"
GOOGLE_TRAIN_ANNOTATIONS_DIR = DATASET + "/Google_Recaptcha_V2_Images_Dataset/split_labels/train/"
GOOGLE_VAL_ANNOTATIONS_DIR = DATASET + "/Google_Recaptcha_V2_Images_Dataset/split_labels/val/"
GOOGLE_TEST_IMG_DIR = DATASET + "/Google_Recaptcha_V2_Images_Dataset/test_img_dir/"
ARCHITECTURE = [
    (32, 3, 1),
    (64, 3, 2),
    ["residual_block", 1],
    (128, 3, 2),
    ["residual_block", 2],
    (256, 3, 2),
    ["residual_block", 8],
    (512, 3, 2),
    ["residual_block", 8],
    (1024, 3, 2),
    ["residual_block", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "scaled_prediction",
    (256, 1, 1),
    "upsample",
    (256, 1, 1),
    (512, 3, 1),
    "scaled_prediction",
    (128, 1, 1),
    "upsample",
    (128, 1, 1),
    (256, 3, 1),
    "scaled_prediction",
]

# Found from k-means algorithm performed on COCO dataset. Should be general 
# enough for Google Dataset as well BUT we should still run k-means here.
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

scale = 1.1
train_transformations = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.Affine(
            translate_percent={"x": 0.1, "y": 0.1},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            shear=(-2,2),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transformations = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

GOOGLE_LABELS = ["Cross", "Chimney", "Stair"]

COCO_LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
"hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
"vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

LABELS_USED = GOOGLE_LABELS