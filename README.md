# Person Detection using YOLOv8

This project demonstrates a basic **person detection system** using **YOLOv8**.
The goal of this project is to understand how an object detection project is created, trained, and tested.

---

## Project Details

- Task: Person Detection
- Model: YOLOv8
- Classes: 1 (person)
- Dataset size: 33 images
- Training: From scratch (no pretrained weights)
- Platform: Windows (CPU)

---

## Folder Structure

person-detection/
│
├── train/
├── valid/
├── data.yaml
├── train_yolo.py
├── live_person_detection.py
├── runs/
├── requirements.txt
└── README.md

yaml
Copy code

---

## What I Did

- Downloaded a person dataset from Roboflow
- Prepared dataset folders (train and validation)
- Configured `data.yaml`
- Trained YOLOv8 model
- Tested the model on images
- Ran live person detection using webcam
- Saved trained weights and results

---

## What I Learned

- Basic understanding of object detection
- How YOLOv8 training works
- Dataset structure for YOLO
- How to run inference on images and live camera
- How to manage an ML project and upload it to GitHub

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
Train the model
bash
Copy code
python train_yolo.py
Live webcam detection
bash
Copy code
python live_person_detection.py
Press q to exit.

Notes
Model accuracy is limited due to small dataset

Project is created for learning purposes

Focus is on project setup, not advanced coding.