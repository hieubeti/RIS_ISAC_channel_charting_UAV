import torch
import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1],128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,3)
)

model.load_state_dict(torch.load("uav_localization_model.pth"))

pred = model(torch.tensor(X,dtype=torch.float32)).detach().numpy()

error = np.linalg.norm(pred-y,axis=1)

print("Mean localization error:",error.mean())
