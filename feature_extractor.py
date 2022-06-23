import torch
import torch.nn.functional as F


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
        self.fc7 = torch.nn.Linear(num_classes, num_classes)
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
        x = x.mean(1)
        x = self.fc7(x)
        return self.soft(x)