## How to Use

This project fine-tunes a custom YOLOv3-based object detection model for Google reCAPTCHA v2-style datasets. Many of the files weren't passed over from the college's server so this repo is not reproducable.

**Swarthmore College CS63: AI Final Project**

<img width="382" height="369" alt="Cross (85)_nms" src="https://github.com/user-attachments/assets/c852ba2e-0e45-48b1-b493-20f6298601a8" />

<img width="369" height="369" alt="Cross (85)_tl" src="https://github.com/user-attachments/assets/d6c3abe4-c681-477f-86e6-bd54daec0295" />

### Setup:

#### 1. Clone the repository

```bash
git clone git@github.com:v6ctor/Yolo-Google-Recaptcha-V2.git
```

#### 2. Install YOLOv5 within repository

YOLOv5 is included as a Git submodule. To correctly initialize everything:

```bash
git clone --recurse-submodules https://github.com/Yolo-Google-Recaptcha-V2/yolov5s.git
```

#### 3. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

#### 4. Train the model

```bash
python3 pipeline.py
```

There is commented code that changes the functionality of the code. It is largely abstracted so that you can train and validate without changing much of the code.
You can change whether you want to pretrain the model or continue training the model (loading from checkpoint) by including
```bash
pipeline.continue_training_model(config.GOOGLE_CHECKPOINT_FILE)
```
If you would like to continue training from a different dataset, change
```
config.GOOGLE_CHECKPOINT_FILE
```
to the dataset of your choice. We include 2 options by default: GOOGLE reCAPTCHA and COCO80.
You'll also need to change the ```config.LABELS_USED``` to the labels being used i.e., COCO80 vs. GOOGLE reCAPTCHA

before
```bash
pipeline.train(optimizer=optimizer, scaler=scaler, scaled_anchors=scaled_anchors)
```
