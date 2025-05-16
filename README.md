# Crowd Detection 

A Python-based AI project that uses a pre-trained object detection model (**Faster R-CNN**) to detect people in a video and identify **crowds** based on spatial proximity and persistence over time.

> A **crowd** is defined as **3 or more persons standing close together for at least 10 consecutive frames**.

---

## Features

- Detects people using **Faster R-CNN (ResNet-50 FPN)** from `torchvision`
- Computes spatial distances between detected persons in each frame
- Tracks persistent close groups across frames
- Logs frame numbers and group sizes where crowd conditions are met
- Exports results to a CSV file

---

## Requirements

- Python 3.8 or above
- PyTorch
- TorchVision
- OpenCV
- Pandas
- SciPy

Install all required packages:

```bash
pip install -r requirements.txt
