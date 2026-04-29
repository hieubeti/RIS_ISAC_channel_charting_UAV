import pandas as pd
import numpy as np

# load dataset
data = pd.read_csv("RIS_ISAC_dataset.csv")

print(data.head())

# tạo CSI complex
data["csi"] = data["csi_real"] + 1j*data["csi_imag"]

# magnitude
data["csi_mag"] = np.abs(data["csi"])

# pivot table
features = data.pivot_table(
    index="uav_id",
    columns="subcarrier",
    values="csi_mag"
)

# label = UAV position
labels = data.groupby("uav_id")[["x","y","z"]].first()

X = features.values
y = labels.values

np.save("X.npy", X)
np.save("y.npy", y)

print("Data processed")
