import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv("data.csv", sep=";")

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(data, mode="train")
val_dataset = ChallengeDataset(data, mode="val")

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=32)

# Create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# In this example, we'll use Binary Cross Entropy with Logits (BCEWithLogitsLoss) and Adam optimizer
criterion = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=model, crit=criterion, optim=optimizer, train_dl=train_loader, val_test_dl=val_loader,cuda=True,early_stopping_patience=100)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=100)



# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
