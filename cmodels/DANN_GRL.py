import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function
import torch.nn.functional as F
from DA.GRL_utils import ReverseLayerGrad

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, delta):
        ctx.delta = delta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.delta
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        x = torch.sigmoid(x)
        return x

class DANN_GRL_VGG16(nn.Module):
    def __init__(self):
        super(DANN_GRL_VGG16, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.cls_fc = nn.Linear(4096, 31)

        self.discriminator = nn.Sequential(
            nn.Linear(25088, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )


    def forward(self, input, delta):
        input = self.features(input)
        features = input.view(input.size(0), -1)
        input = self.classifier(features)
        reverse_features = ReverseLayerGrad.apply(features, delta)

        class_pred = self.cls_fc(input)
        loss_adv = self.discriminator(reverse_features)

        return class_pred, loss_adv

class DANN_GRL_Resnet(nn.Module):
    def __init__(self, resnet_func, pretrained,  num_classes):
        super(DANN_GRL_Resnet, self).__init__()

        self.sharedNet = resnet_func(pretrained)
        self.domain_classifier = Discriminator(input_dim=self.sharedNet.fc.in_features, hidden_dim=4096)
        self.cls_fc = nn.Linear(self.sharedNet.fc.in_features, num_classes)

    def forward(self, input, delta=1, source=True):
        input = self.sharedNet(input)
        features = input.view(input.size(0), -1)

        class_output = self.cls_fc(features)
        loss_adv = self.get_adversarial_result(
            features, source, delta)

        return class_output, loss_adv

    def get_adversarial_result(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def nforward(self, source):
        source = self.sharedNet(source)
        source = source.view(source.size(0), -1)
        source = self.cls_fc(source)
        return source

class DANN_GRL_Alexnet(nn.Module):

    def __init__(self, alexnet_func, pretrained, num_classes=31):
        super(DANN_GRL_Alexnet, self).__init__()
        self.alexnet = alexnet_func(pretrained)
        self.domain_classifier = Discriminator(input_dim=256 * 6 * 6, hidden_dim=4096)
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

    def get_adversarial_result(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def nforward(self, x):
        x = self.alexnet(x) #Same as self.SharedNet(x)
        x = x.view(x.size(0), -1)
        return self.cls_fc(x)

    def forward(self, input, delta=1, source=True):
        features = self.alexnet(input)
        features = features.view(input.size(0), -1)

        class_output = self.cls_fc(features)
        loss_adv = self.get_adversarial_result(
            features, source, delta)


        return class_output, loss_adv

class DANN_GRL_LeNet(nn.Module):

    def __init__(self, lenet_func, pretrained, num_classes=10., input_size=28):
        super(DANN_GRL_LeNet, self).__init__()
        self.lenet = lenet_func(pretrained, num_classes, input_size)
        self.domain_classifier = Discriminator(input_dim=self.lenet.classifier[0].in_features, hidden_dim=4096)
        self.cls_fc = nn.Sequential(
            nn.Linear(self.lenet.classifier[0].in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def features(self, x):
        return self.alexnet(x)

    def get_adversarial_result(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def nforward(self, x):
        x = self.lenet(x) #Same as self.SharedNet(x)
        x = x.view(x.size(0), -1)
        return self.cls_fc(x)

    def forward(self, input, delta=1, source=True):
        features = self.lenet(input)
        features = features.view(input.size(0), -1)

        class_output = self.cls_fc(features)
        loss_adv = self.get_adversarial_result(
            features, source, delta)

        return class_output, loss_adv

class DANNMnistModel(nn.Module):

    def __init__(self):
        super(DANNMnistModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, delta):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerGrad.apply(feature, delta)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output