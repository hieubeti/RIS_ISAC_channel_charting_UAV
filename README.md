# Channel Charting for UAV Navigation in RIS-Assisted ISAC Systems

This repository contains a Python-based simulation project for UAV localization in a Reconfigurable Intelligent Surface (RIS)-assisted Integrated Sensing and Communication (ISAC) system.

The project focuses on generating wireless channel data, preprocessing Channel State Information (CSI), applying channel charting, and training a machine learning model to estimate UAV positions.

## Project Overview

Reliable UAV navigation is difficult in GNSS-denied environments such as urban canyons, indoor spaces, and complex low-altitude scenarios. This project uses wireless channel information instead of relying only on GPS/GNSS.

The main idea is that CSI contains spatial information related to the UAV position. By learning the relationship between CSI and UAV coordinates, the system can estimate the UAV location using communication signals.

## Main Features

- RIS-assisted ISAC simulation
- UAV localization using wireless channel data
- CSI preprocessing and feature extraction
- Channel charting for low-dimensional representation
- Machine learning-based localization model
- Training and evaluation pipeline
- Dataset support using CSV format
- Blender environment support for 3D simulation

## Repository Structure

```text
.
├── Blender_code/
│   └── Blender environment scripts
│
├── README.md
├── RIS_ISAC_dataset.csv
├── channel_charting.py
├── localization_model.py
├── main.py
├── preprocess.py
└── train_model.py
```

## File Description

| File / Folder | Description |
|---|---|
| `Blender_code/` | Contains Blender scripts used to create or modify the 3D simulation environment. |
| `RIS_ISAC_dataset.csv` | Dataset containing simulated RIS-assisted ISAC channel features and UAV position labels. |
| `preprocess.py` | Loads, cleans, normalizes, and splits the dataset. |
| `channel_charting.py` | Applies channel charting or dimensionality reduction to CSI features. |
| `localization_model.py` | Defines the machine learning model for UAV position prediction. |
| `train_model.py` | Trains the localization model and saves training results. |
| `main.py` | Runs the full pipeline from preprocessing to evaluation. |
| `README.md` | Documentation for the project. |

## System Model

The simulated system contains:

- Multiple Base Stations (BSs)
- Multiple Reconfigurable Intelligent Surfaces (RISs)
- One UAV receiver
- Wireless channel measurements
- UAV position labels
- RIS-assisted reflected paths
- Line-of-Sight and Non-Line-of-Sight propagation components

The UAV receives pilot signals from base stations. The channel response is used as input data for localization.

## Simulation Pipeline

```text
3D Environment Design
        ↓
BS / RIS / UAV Deployment
        ↓
Wireless Channel Data Generation
        ↓
CSI Dataset Construction
        ↓
Data Preprocessing
        ↓
Channel Charting
        ↓
Localization Model Training
        ↓
UAV Position Estimation
        ↓
Performance Evaluation
```

## Dataset Format

The dataset file is:

```text
RIS_ISAC_dataset.csv
```

A typical dataset format is:

```text
feature_1, feature_2, feature_3, ..., feature_n, x, y
```

Where:

- `feature_1 ... feature_n` are wireless channel or CSI-related features.
- `x` and `y` are UAV position coordinates.
- Optional columns may include altitude, SNR, BS index, RIS index, or scenario type.

Example:

```csv
csi_1,csi_2,csi_3,csi_4,x,y
0.245,0.813,0.124,0.551,20.5,34.2
0.231,0.802,0.118,0.547,21.0,34.5
```

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

If your implementation uses deep learning, install one of the following:

For PyTorch:

```bash
pip install torch torchvision torchaudio
```

For TensorFlow:

```bash
pip install tensorflow
```

For UMAP-based channel charting:

```bash
pip install umap-learn
```

## How to Run

### Run the Full Pipeline

```bash
python main.py
```

This command performs:

1. Dataset loading
2. Data preprocessing
3. Channel charting
4. Model training
5. UAV position prediction
6. Performance evaluation

### Run Preprocessing Only

```bash
python preprocess.py
```

### Run Channel Charting Only

```bash
python channel_charting.py
```

### Train the Localization Model

```bash
python train_model.py
```

## Methodology

### 1. Environment Generation

The 3D simulation environment is created using Blender. The environment may contain buildings, roads, base stations, RIS panels, and UAV flight paths.

The environment is used to support wireless channel simulation and dataset generation.

### 2. CSI Data Processing

The raw wireless channel data is processed before training. Preprocessing may include:

- Removing invalid samples
- Normalizing channel features
- Separating input features and UAV labels
- Splitting training and testing data
- Converting CSI into a more useful feature representation

### 3. Channel Charting

Channel charting maps high-dimensional CSI data into a low-dimensional space while preserving spatial relationships.

The goal is:

```text
Nearby UAV positions should have nearby channel chart representations.
```

This helps the model learn the relationship between wireless channels and physical UAV locations.

### 4. Localization Model

The localization model learns the mapping:

```text
CSI Features → UAV Coordinates
```

The output of the model is the predicted UAV position:

```text
Predicted Position = (x, y)
```

### 5. Evaluation

The localization performance can be evaluated using:

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- Localization Distance Error
- CE90 / 90th Percentile Error
- Channel chart visualization

## Example Output

After training, the program may display:

```text
Training completed.
Mean Absolute Error: 3.66 m
Root Mean Squared Error: 4.21 m
90th Percentile Error: 5.71 m
```

The project may also generate visual outputs such as:

- Ground-truth UAV trajectory
- Predicted UAV trajectory
- Channel chart plot
- Training loss curve
- Localization error distribution

## Expected Results

The expected result is that the trained model can estimate UAV positions from wireless channel data.

RIS deployment is expected to improve localization performance by creating additional reflected paths and increasing channel diversity.

## Update History

```text
Blender_code
Update Blender_code

README.md
Update README.md

RIS_ISAC_dataset.csv
Add files via upload

channel_charting.py
Create channel_charting.py

localization_model.py
Create localization_model.py

main.py
Create main.py

preprocess.py
Create preprocess.py

train_model.py
Update and rename train_modal.py to train_model.py
```

## Future Work

Possible improvements include:

- Add Sionna ray-tracing support
- Add LoS and NLoS scenario comparison
- Add RIS phase shift optimization
- Add semi-supervised learning
- Add Siamese neural network-based channel charting
- Add UAV trajectory visualization in 3D
- Add SNR-based performance comparison
- Add support for real-world CSI datasets

## Reference

This project is inspired by research on channel charting for UAV navigation in RIS-assisted ISAC systems.

## License

This project is for research and educational purposes.
