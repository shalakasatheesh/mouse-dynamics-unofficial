from move_dataset import MoveDataset, Normalize
from feature_extractor import FeatureExtractor
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

dataset = MoveDataset(pickle_file='move_data.pkl', transform=Normalize())
loader = DataLoader(dataset, batch_size=8, shuffle=True)
writer = SummaryWriter()

model = FeatureExtractor(2, dataset.unique_user_count(), 128)
print(summary(model, (2, 128), device='cpu'))
writer.add_graph(model, torch.rand(1, 2, 128))
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
total_length = len(loader)
for epoch in range(100):
    print('Running epoch #', epoch, '...')
    i = 0
    for batch, targets in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch * total_length + i)
        print(' Epoch', epoch + 1, ' - Batch #', i, '/', int(len(dataset) / 8.0), 'Loss:', loss.item())
        i += 1

        if i % 50 == 0:
            writer.flush()

writer.close()
