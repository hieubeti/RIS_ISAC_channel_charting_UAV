import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

print("=================================")
print("RIS-ISAC Channel Chart Pipeline")
print("=================================")

# -------------------------------------------------
# STEP 1: LOAD DATASET
# -------------------------------------------------

print("\nStep 1: Preprocessing CSI dataset...")

data = pd.read_csv("RIS_ISAC_dataset.csv")

print(data.head())

# tạo CSI complex
data["csi"] = data["csi_real"] + 1j * data["csi_imag"]

# magnitude
data["csi_mag"] = np.abs(data["csi"])

print("Data processed")

# -------------------------------------------------
# STEP 2: CREATE FEATURES PER UAV
# -------------------------------------------------

print("\nStep 2: Generating Channel Chart...")

grouped = data.groupby("uav_id")

X = []
positions = []

for uid, g in grouped:

    feature = g["csi_mag"].values
    X.append(feature)

    pos = g[["x","y"]].iloc[0].values
    positions.append(pos)

X = np.array(X)
positions = np.array(positions)

print("Feature matrix shape:", X.shape)

# -------------------------------------------------
# STEP 3: CHANNEL CHART (PCA)
# -------------------------------------------------

print("Generating Channel Chart...")

pca = PCA(n_components=2)

chart = pca.fit_transform(X)

print("Channel chart created")

# -------------------------------------------------
# STEP 4: UAV LOCALIZATION MODEL
# -------------------------------------------------

print("\nStep 3: Estimating UAV trajectory...")

X_train, X_test, y_train, y_test = train_test_split(
    chart, positions, test_size=0.2, random_state=42
)

model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Trajectory estimation finished")

# -------------------------------------------------
# STEP 5: LOCALIZATION ERROR
# -------------------------------------------------

error = np.linalg.norm(y_pred - y_test, axis=1)

mean_error = np.mean(error)

print("\nMean localization error:", round(mean_error,2), "meters")

print("\n=================================")
print("Pipeline Finished Successfully")
print("=================================")

# -------------------------------------------------
# STEP 6: PLOT RESULTS
# -------------------------------------------------

print("Plotting results...")

# ==============================
# Channel Chart
# ==============================

plt.figure()

plt.scatter(chart[:,0], chart[:,1], c=positions[:,0], cmap="viridis")

plt.title("Channel Chart")

plt.xlabel("Chart Dimension 1")
plt.ylabel("Chart Dimension 2")

plt.colorbar(label="UAV X position")

plt.savefig("channel_chart.png")


# ==============================
# True vs Predicted UAV
# ==============================

plt.figure()

plt.scatter(y_test[:,0], y_test[:,1], label="True UAV", marker="o")

plt.scatter(y_pred[:,0], y_pred[:,1], label="Predicted UAV", marker="x")

plt.legend()

plt.title("UAV Localization")

plt.xlabel("X position")
plt.ylabel("Y position")

plt.savefig("uav_localization.png")


# ==============================
# UAV Trajectory Comparison
# ==============================

# UAV trajectory thật
true_traj = positions

# UAV trajectory dự đoán
pred_traj = model.predict(chart)

plt.figure()

plt.plot(true_traj[:,0], true_traj[:,1], label="True UAV Trajectory")

plt.plot(pred_traj[:,0], pred_traj[:,1], label="Predicted UAV Trajectory")

plt.legend()

plt.title("UAV Trajectory Comparison")

plt.xlabel("X position")
plt.ylabel("Y position")

plt.savefig("uav_trajectory.png")


# ==============================
# Error histogram
# ==============================

plt.figure()

plt.hist(error, bins=20)

plt.title("Localization Error Distribution")

plt.xlabel("Error (meters)")
plt.ylabel("Count")

plt.savefig("localization_error.png")

plt.show()

print("\nCharts saved:")
print("channel_chart.png")
print("uav_localization.png")
print("uav_trajectory.png")
print("localization_error.png")
