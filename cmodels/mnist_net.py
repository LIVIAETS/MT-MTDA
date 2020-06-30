import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output = self.features(img)
        output = output.view(img.size(0), self.classifier[0].in_features)
        output = self.classifier(output)
        return output

class S_LeNet5(nn.Module):

    def __init__(self):
        super(S_LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 60, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(60, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        output = self.features(img)
        output = output.view(img.size(0), self.classifier[0].in_features)
        output = self.classifier(output)
        return output