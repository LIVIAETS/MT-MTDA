import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self, pretrained=False, num_classes=10, input_size=28):
        super(LeNet, self).__init__()
        if input_size == 32:
            in_features = 1250
        elif input_size == 28:
            in_features = 800
        elif input_size == 227:
            in_features = 140450

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

class MTDA_ITA_classifier(nn.Module):

    def __init__(self, pretrained=False, num_classes=10, input_size=32):
        super(MTDA_ITA_classifier, self).__init__()

        if input_size == 32:
            in_features = 484
        elif input_size == 227:
            in_features = 45796

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x