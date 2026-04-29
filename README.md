Project Objective

The goal of this project is to estimate the position of a UAV using wireless channel information in a RIS-assisted ISAC environment.

Instead of relying only on GPS/GNSS, this project uses Channel State Information (CSI) or channel-related features collected from the communication system. These features are processed and mapped to UAV coordinates using channel charting and machine learning.

Background

In UAV navigation, traditional GNSS-based localization may become unreliable in urban, indoor, or obstructed environments. RIS-assisted ISAC systems can improve wireless propagation by introducing controllable reflected paths.

This project follows the idea that wireless channel features are strongly related to physical UAV positions. By learning this relationship, the system can estimate UAV locations from channel measurements.

Simulation Pipeline

The complete simulation workflow is:

3D Environment Design
        ↓
RIS / BS / UAV Deployment
        ↓
Channel Data Generation
        ↓
Dataset Construction
        ↓
Data Preprocessing
        ↓
Channel Charting
        ↓
Localization Model Training
        ↓
Position Estimation and Evaluation
Dataset

The dataset file is:

RIS_ISAC_dataset.csv

The dataset is expected to contain simulated wireless channel features and UAV coordinate labels.

Example structure:

feature_1, feature_2, feature_3, ..., x, y

Where:

feature_1 ... feature_n are channel-related features.
x and y are UAV position coordinates.
Optional columns may include altitude, BS index, RIS index, SNR, or scenario type.
Requirements

Install the required Python libraries:

pip install numpy pandas matplotlib scikit-learn tensorflow torch

Depending on your implementation, you may also need:

pip install umap-learn

If the project uses PyTorch only:

pip install torch torchvision torchaudio

If the project uses TensorFlow only:

pip install tensorflow
How to Run
1. Clone the Repository
git clone <your-repository-link>
cd <your-repository-name>
2. Check the Dataset

Make sure the dataset exists in the project folder:

RIS_ISAC_dataset.csv
3. Run the Main Program
python main.py

This will run the full pipeline, including:

Loading the dataset
Preprocessing the channel data
Applying channel charting
Training the localization model
Evaluating localization accuracy
4. Train the Model Separately
python train_model.py
5. Run Preprocessing Only
python preprocess.py
6. Run Channel Charting Only
python channel_charting.py
Main Components
1. Blender Environment Generation

The Blender_code/ folder is used to build or export the 3D wireless environment.

This environment may include:

Buildings
UAV flight trajectory
Base stations
RIS panels
Coordinate system
Simulation area

The generated environment can be used for ray-tracing-based wireless channel simulation.

2. Preprocessing

The preprocessing stage prepares the raw dataset before training.

Typical preprocessing steps include:

Loading CSV data
Removing invalid values
Separating input features and coordinate labels
Normalizing channel features
Splitting data into training and testing sets
3. Channel Charting

Channel charting maps high-dimensional wireless channel data into a low-dimensional representation.

The purpose is to preserve the spatial relationship between UAV positions. If two UAV positions are close in physical space, their channel chart representations should also be close.

4. Localization Model

The localization model learns the mapping:

Channel Features → UAV Position

The model predicts UAV coordinates from the processed channel data.

Example output:

Predicted position: x = 25.3 m, y = 41.8 m
5. Training and Evaluation

The model is trained using labeled UAV position data.

Common evaluation metrics include:

Mean Absolute Error
Mean Squared Error
Root Mean Squared Error
Localization distance error
90th percentile positioning error
Example Output

After running the project, the program may output results such as:

Training completed.
Mean Absolute Error: 3.66 m
Root Mean Squared Error: 4.21 m

The system may also generate plots such as:

UAV ground-truth trajectory
Predicted UAV trajectory
Channel chart visualization
Training loss curve
Localization error distribution
Expected Results

The project is expected to show that RIS-assisted channel information can improve UAV localization performance. Channel charting helps reduce the dependency on large amounts of labeled position data by learning the geometric structure of wireless channel measurements.

Repository Update History
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
Notes
The dataset is simulation-based.
The localization accuracy depends on the quality of the generated channel data.
The Blender environment and RIS/BS placement should be consistent with the dataset.
If using real ray tracing tools, make sure the exported environment and coordinate system are correctly aligned.
Future Improvements

Possible improvements include:

Adding Sionna ray tracing support
Adding RIS phase optimization
Supporting LoS and NLoS scenario comparison
Adding semi-supervised training
Adding Siamese neural network-based channel charting
Evaluating localization under different SNR values
Visualizing UAV trajectory in 2D and 3D
