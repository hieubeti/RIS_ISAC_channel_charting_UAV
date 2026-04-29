import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Generating Channel Chart...")

# Load processed dataset
data = pd.read_csv("RIS_ISAC_dataset.csv")

# Create CSI complex value
data["csi"] = data["csi_real"] + 1j * data["csi_imag"]

# Group by UAV position
grouped = data.groupby("uav_id")

features = []

for _, group in grouped:
    csi_vector = group["csi"].values
    features.append(np.abs(csi_vector))

features = np.array(features)

# PCA to 2D
pca = PCA(n_components=2)
chart = pca.fit_transform(features)

# Plot
plt.scatter(chart[:,0], chart[:,1])
plt.title("Channel Chart")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.savefig("channel_chart.png")
plt.show()

print("Channel chart created")
