# MoCE-ViT: Contrastive Enhancement Vision Transformer for UAV-Based Paddy Crop Health Classification

This repository provides the source code for **MoCE-ViT**, a deep learning pipeline designed for classifying paddy crop health using RGB imagery captured by UAVs. The model combines a **learnable image enhancement block**, a **dual-branch Vision Transformer (ViT)** architecture, and **momentum contrastive learning (MoCo)** to achieve robust classification under diverse lighting and field conditions.

---

## Dataset

The dataset used is available on IEEE DataPort:

 [Paddy Crop RGB Drone Data] [doi:10.21227/jzw5-hk19](https://ieee-dataport.org/documents/paddy-crop-rgb-drone-data)) 
- RGB images (224×224 px) labeled as healthy or unhealthy
- Captured using the **DJI Mavic 3M RGB sensor**
- Fields located at IIT Kharagpur and nearby agricultural plots

---

##  Requirements

Install dependencies using:


pip install -r requirements.txt


**Python ≥ 3.8** and **CUDA-enabled GPU** are recommended for training.

---

##  How to Run

### Train and Evaluate

python main.py


This runs:
- `train.py → run_training()`
- `eval.py → evaluate_external()`

All outputs and logs will be stored in the configured output directory.

---

##  Project Structure

```
├── main.py                 # Entry point
├── model.py                # MoCE-ViT: model, enhancement, contrastive loss
├── train.py                # Training loop
├── eval.py             # Evaluation on external UAV datasets
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore 
```

---

##  Citation

> This work has been submitted to the IEEE Transactions on Image Processing (TIP).  
> Title: *MoCE-ViT: Momentum-Contrastive Enhanced Vision Transformer for UAV-Based Crop Health Classification *

Please cite the dataset and this repository if used in research or publications.

---

## Contact

For questions or collaborations, please contact the author via GitHub Issues or email.
