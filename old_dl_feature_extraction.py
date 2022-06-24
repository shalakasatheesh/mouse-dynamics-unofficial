import torch
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

user_path = 'DLRV_Dataset/user_df_final_normalized'
actions = ['scroll', 'click', 'move']
scroll = pd.DataFrame()
move = pd.DataFrame()
click = pd.DataFrame()

for action in actions:
    path = listdir(user_path + '/' + action)
    for file in path:
        df = pd.read_pickle(user_path + '/' + action + '/' + file)
        if action == 'click':
            click = pd.concat([click, df])
        elif action == 'move':
            move = pd.concat([click, df])
        elif action == 'scroll':
            scroll = pd.concat([click, df])

# Split into data (X) and labels (Y))
data = move[['dt', 'dx', 'dy', 'token']]


class FeatureExtractor(torch.nn.Module):

    def __init__(self, num_features, num_classes, num_filters):
        super(FeatureExtractor, self).__init__()

        self.fc1 = torch.nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=8, padding='same')
        self.fc2 = torch.nn.BatchNorm1d(num_filters)

        self.fc3 = torch.nn.Conv1d(num_filters, 2 * num_filters, kernel_size=5, padding='same')
        self.fc4 = torch.nn.BatchNorm1d(2 * num_filters)

        self.fc5 = torch.nn.Conv1d(2 * num_filters, num_filters, kernel_size=3, padding='same')
        self.fc6 = torch.nn.BatchNorm1d(num_filters)

        self.gap = torch.nn.AdaptiveAvgPool1d(num_classes)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = self.fc6(x)
        x = F.relu(x)

        x = self.gap(x)
        return self.soft(x)


grouped_by = data.groupby(['token'])

user_ids = {}
losses = []

cleaned_data = {}


def assign_user_ids(data):
    row = 0
    for (token, df) in data:
        blocks = df.head(128)
        if len(blocks) == 128:
            user_ids[token] = row
            row = row + 1
            cleaned_data[token] = (blocks['dt'], blocks['dx'], blocks['dy'])


assign_user_ids(grouped_by)

model = torch.nn.Sequential(
    torch.nn.Conv1d(2, 128, kernel_size=8, padding='same'),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(),
    torch.nn.Conv1d(128, 256, kernel_size=5, padding='same'),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Conv1d(256, 128, kernel_size=3, padding='same'),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool1d(len(cleaned_data)),
    torch.nn.Linear(len(cleaned_data), len(cleaned_data)),
    torch.nn.Softmax(dim=1)
)

print(summary(model, (2, 128), device='cpu'))

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
X = np.empty(shape=(len(cleaned_data), 2, 128))
y = np.zeros(shape=(len(cleaned_data), len(user_ids)))
row_x = 0
for (token, data) in cleaned_data.items():
    y[row_x, user_ids[token]] = 1.0
    for row in range(128):
        X[row_x, 0, row] = data[1].iloc[row]
        X[row_x, 1, row] = data[2].iloc[row]
    row_x += 1


def test_data():
    output = model(torch.Tensor(X))
    print(output.shape)
    for row in range(output.shape[0]):
        entry = output[row]



for epoch in range(250):
    optimizer.zero_grad()
    output = model(torch.Tensor(X))
    loss = F.cross_entropy(output, torch.Tensor(y).long())
    loss.backward()
    optimizer.step()
    losses.append((epoch, loss.item()))
    print(loss.item())
    test_data()

plt.figure()
plt.plot([x[0] for x in losses], [x[1] for x in losses])
plt.show()
