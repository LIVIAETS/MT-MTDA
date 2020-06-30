import torch
import torch.nn as nn
import DA.mmd as mmd
import torchvision.models as models

class DANNetVGG16(nn.Module):
    def __init__(self, vgg_func, pretrained, num_classes=31):
        super(DANNetVGG16, self).__init__()
        model = vgg_func(pretrained=pretrained)  #False

        self.features = model.features


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.cls_fc = nn.Linear(4096, num_classes)

    def forward(self, source, target):
        loss = torch.FloatTensor([0]).to(source.device)
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)

        if self.training == True:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        return source, loss, target

    def nforward(self, source):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        source = self.cls_fc(source)
        return source

    def s_forward(self, source):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        source = self.cls_fc(source)
        return source

class DANNet_ResNet(nn.Module):

    def __init__(self, resnet_func, pretrained, num_classes=31):
        super(DANNet_ResNet, self).__init__()
        self.sharedNet = resnet_func(pretrained)
        self.cls_fc = nn.Linear(self.sharedNet.fc.in_features, num_classes)

    def features(self, x):
        return self.sharedNet.features(x)

    def nforward(self, x):
        x = self.features(x) #Same as self.SharedNet(x)
        return self.cls_fc(x)

    def forward(self, source, target):
        #Hack for Multiple GPU...
        loss = torch.zeros(1, device=source.device)
        source = self.sharedNet(source)

        if self.training == True:
            target = self.sharedNet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss = mmd.mmd_rbf_noaccelerate(source, target)
            target = self.cls_fc(target)

        source = self.cls_fc(source)

        return source, loss, target

class DANNet_Alexnet(nn.Module):

    def __init__(self, alexnet_func, pretrained, num_classes=31):
        super(DANNet_Alexnet, self).__init__()
        self.alexnet = alexnet_func(pretrained)
        self.cls_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def features(self, x):
        return self.alexnet(x)

    def nforward(self, x):
        x = self.alexnet(x) #Same as self.SharedNet(x)
        return self.cls_fc(x)

    def forward(self, source, target):
        #Hack for Multiple GPU...
        loss = torch.zeros(1, device=source.device)
        source = self.alexnet(source)

        if self.training == True:
            target = self.alexnet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss = mmd.mmd_rbf_noaccelerate(source, target)
            target = self.cls_fc(target)

        source = self.cls_fc(source)

        return source, loss, target

class DANNet_Effnet(nn.Module):

    def __init__(self, alexnet_func, pretrained, num_classes=31):
        super(DANNet_Alexnet, self).__init__()
        self.alexnet = alexnet_func(pretrained)
        self.cls_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def features(self, x):
        return self.alexnet(x)

    def nforward(self, x):
        x = self.alexnet(x) #Same as self.SharedNet(x)
        return self.cls_fc(x)

    def forward(self, source, target):
        #Hack for Multiple GPU...
        loss = torch.zeros(1, device=source.device)
        source = self.alexnet(source)

        if self.training == True:
            target = self.alexnet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss = mmd.mmd_rbf_noaccelerate(source, target)
            target = self.cls_fc(target)

        source = self.cls_fc(source)

        return source, loss, target