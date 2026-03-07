import torch
import torch.nn as nn
import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = nn.Sequential(

    nn.Linear(X.shape[1],128),
    nn.ReLU(),

    nn.Linear(128,64),
    nn.ReLU(),

    nn.Linear(64,3)

)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):

    pred = model(X)

    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch, loss.item())

torch.save(model.state_dict(),"uav_localization_model.pth")

print("Training finished")
