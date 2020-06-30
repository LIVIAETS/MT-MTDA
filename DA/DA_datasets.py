import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data
import torch
from PIL import Image
import os
import numpy as np
import DA.MNISTM as MNISTM

def get_digit_five_train_loader(d_name, batch_size=16, num_workers=1, pin_memory=False, drop_last=False, resize=28):

    d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                        transforms.ToTensor(),
                                        transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ])

    if d_name == "MNIST":
        d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = datasets.MNIST('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = datasets.MNIST('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == "MNIST-M":
        d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == 'SVHN':
        trainset = datasets.SVHN('./digits_data/', download=True, split="train", transform=d_transforms)
        valset = datasets.SVHN('./digits_data/', download=True, split="test", transform=d_transforms)
    elif d_name == 'USPS':
        d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = datasets.USPS('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = datasets.USPS('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == 'SY':
        trainset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_train', transform=d_transforms)
        valset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_valid', transform=d_transforms)

    trainset.num_classes = 10
    valset.num_classes = 10

    train_loader = data.DataLoader(trainset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last)

    test_loader = data.DataLoader(valset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last)

    return train_loader, test_loader

def get_digit_five_train_no_split_loader(d_name, batch_size=16, num_workers=1, pin_memory=False, drop_last=False, resize=28):

    d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                        transforms.ToTensor(),
                                        transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ])

    if d_name == "MNIST":
        d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = datasets.MNIST('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = datasets.MNIST('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == "MNIST-M":
        d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == 'SVHN':
        trainset = datasets.SVHN('./digits_data/', download=True, split="train", transform=d_transforms)
        valset = datasets.SVHN('./digits_data/', download=True, split="test", transform=d_transforms)
    elif d_name == 'USPS':
        d_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                           transforms.Grayscale(3),
                                           transforms.ToTensor(),
                                           transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
        trainset = datasets.USPS('./digits_data/', download=True, train=True, transform=d_transforms)
        valset = datasets.USPS('./digits_data/', download=True, train=False, transform=d_transforms)
    elif d_name == 'SY':
        trainset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_train', transform=d_transforms)
        valset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_valid', transform=d_transforms)

    all_set = torch.utils.data.ConcatDataset([trainset, valset])
    all_set.num_classes = 10

    return data.DataLoader(all_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def get_digits_loaders_concat(targets, batch_size=16, num_workers=1, pin_memory=False, drop_last=False, resize=28):

    d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                        transforms.ToTensor(),
                                        transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ])
    vals = []
    trains = []
    for d_name in targets:
        if d_name == "MNIST":
            d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                                transforms.Grayscale(3),
                                                transforms.ToTensor(),
                                                transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ])
            trainset = datasets.MNIST('./digits_data/', download=True, train=True, transform=d_transforms)
            valset = datasets.MNIST('./digits_data/', download=True, train=False, transform=d_transforms)
        elif d_name == "MNIST-M":
            d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                                transforms.Grayscale(3),
                                                transforms.ToTensor(),
                                                transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ])
            trainset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=True, transform=d_transforms)
            valset = MNISTM.MNISTM_dataset('./digits_data/', download=True, train=False, transform=d_transforms)
        elif d_name == 'SVHN':
            trainset = datasets.SVHN('./digits_data/', download=True, split="train", transform=d_transforms)
            valset = datasets.SVHN('./digits_data/', download=True, split="test", transform=d_transforms)
        elif d_name == 'USPS':
            d_transforms = transforms.Compose([transforms.Resize((resize,resize)),
                                                transforms.Grayscale(3),
                                                transforms.ToTensor(),
                                                transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ])
            trainset = datasets.USPS('./digits_data/', download=True, train=True, transform=d_transforms)
            valset = datasets.USPS('./digits_data/', download=True, train=False, transform=d_transforms)
        elif d_name == 'SY':
            trainset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_train', transform=d_transforms)
            valset = datasets.ImageFolder('./digits_data/synthetic_digits/imgs_valid', transform=d_transforms)
        trains.append(trainset)
        vals.append(valset)

    train_sets = torch.utils.data.ConcatDataset(trains)
    val_sets = torch.utils.data.ConcatDataset(vals)
    train_sets.num_classes = 10
    val_sets.num_classes = 10

    trainloader = data.DataLoader(train_sets,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

    valloader = data.DataLoader(val_sets,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

    return trainloader, valloader


def get_pacs_target_loader(data_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(data_path,img_transform)
    dataset.num_classes = 7

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def get_source_target_loader(dataset_name, source_path, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader, target_dataloader, target_testloader

def get_m_source_target_loader(dataset_name, source_path_1, source_path_2, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader_1 = None
    source_dataloader_2 = None
    if dataset_name == "Office31":
        if source_path_1 != "":
            source_dataloader_1 = office_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = office_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path_1 != "":
            source_dataloader_1 = imageclef_train_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = imageclef_train_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader_1, source_dataloader_2, target_dataloader, target_testloader

def get_source_m_target_loader(dataset_name, source_path, target_path_s, batch_size=16, num_workers=1, pin_memory=False, drop_last=False, **kwargs):

    targets_dataloader = [0] * len(target_path_s)
    targets_testloader = [0] * len(target_path_s)

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        for i, p in enumerate(target_path_s):
            targets_dataloader[i] = office_loader(p, batch_size, num_workers, pin_memory, drop_last)
            targets_testloader[i] = office_test_loader(p, batch_size, num_workers, pin_memory)
    elif dataset_name == "OfficeHome":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        for i, p in enumerate(target_path_s):
            targets_dataloader[i] = office_loader(p, batch_size, num_workers, pin_memory, drop_last)
            targets_testloader[i] = office_test_loader(p, batch_size, num_workers, pin_memory)
    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        for i, p in enumerate(target_path_s):
            targets_dataloader[i] = imageclef_train_loader(p, batch_size, num_workers, pin_memory, drop_last)
            targets_testloader[i] = imageclef_test_loader(p, batch_size, num_workers, pin_memory)
    elif dataset_name == "Digits":
        source_dataloader, _ = get_digit_five_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last, **kwargs)
        for i, t in enumerate(target_path_s):
            targets_dataloader[i], targets_testloader[i] = get_digit_five_train_loader(t, batch_size, num_workers, pin_memory,
                                                               drop_last, **kwargs)
    elif dataset_name == "Digits_no_split":
        source_dataloader= get_digit_five_train_no_split_loader(source_path, batch_size, num_workers, pin_memory, drop_last, **kwargs)
        for i, t in enumerate(target_path_s):
            targets_dataloader[i] = get_digit_five_train_no_split_loader(t, batch_size, num_workers, pin_memory,
                                                               drop_last, **kwargs)
            targets_testloader[i] = get_digit_five_train_no_split_loader(t, batch_size, num_workers,
                                                                         pin_memory,
                                                                         drop_last, **kwargs)
    elif dataset_name == "PACS":
        source_dataloader = get_pacs_target_loader(source_path, batch_size, num_workers, pin_memory,
                                                                 drop_last)
        for i, t in enumerate(target_path_s):
            targets_dataloader[i] = get_pacs_target_loader(t, batch_size, num_workers,
                                                                         pin_memory,
                                                                         drop_last)
            targets_testloader[i] = get_pacs_target_loader(t, batch_size, num_workers,
                                                                         pin_memory,
                                                                         drop_last)
    else:
        raise("Dataset not handled")

    return source_dataloader, targets_dataloader, targets_testloader

def get_source_concat_target_loader(dataset_name, source_path, target_paths, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_m_concat_loader(target_paths, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_m_concat_test_loader(target_paths, batch_size, num_workers, pin_memory)
    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        raise ("To be implemented")
    elif dataset_name == "OfficeHome":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_m_concat_loader(target_paths, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_m_concat_test_loader(target_paths, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")


    return source_dataloader, target_dataloader, target_testloader

def get_train_test_loader(dataset_name, data_path, batch_size=16, num_workers=1, pin_memory=False):
    if dataset_name == "Office31":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)


    elif dataset_name == "ImageClef":
        train_loader, test_loader = imageclef_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return train_loader, test_loader

def office_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    train_set, test_set = data.random_split(dataset,
                                            [int(0.7 * dataset.__len__()), dataset.__len__() - int(0.7 * dataset.__len__())])

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def office_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset.num_classes = len(dataset.classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def office_m_concat_loader(paths, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    datasetsF = []
    for p in paths:
        datasetsF.append(datasets.ImageFolder(p,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                       ])))
    dataset = torch.utils.data.ConcatDataset(datasetsF)
    dataset.num_classes = len(datasetsF[0].classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def office_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def office_m_concat_test_loader(target_paths, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    datasetsF = []

    for p in target_paths:
        datasetsF.append(datasets.ImageFolder(p,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ])))

    dataset = torch.utils.data.ConcatDataset(datasetsF)
    dataset.num_classes = len(datasetsF[0].classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def imageclef_train_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def imageclef_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
                       ]))

    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def imageclef_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CLEFImage(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    train_set, test_set = data.random_split(dataset,
                                            [int(0.7 * dataset.__len__()), dataset.__len__() - int(0.7 * dataset.__len__())])

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_imageclef_dataset(path):
    images = []
    dataset_name = path.split("/")[-1]
    label_path = os.path.join(path, ".." , "list", "{}List.txt".format(dataset_name))
    image_folder = os.path.join(path, "..", "{}".format(dataset_name))
    labeltxt = open(label_path)
    for line in labeltxt:
        pre_path, label = line.strip().split(' ')
        image_name = pre_path.split("/")[-1]
        image_path = os.path.join(image_folder, image_name)

        gt = int(label)
        item = (image_path, gt)
        images.append(item)
    return images

class CLEFImage(data.Dataset):
    def __init__(self, root, transform=None, image_loader=default_loader):
        imgs = make_imageclef_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.image_loader = image_loader
        self.num_classes = 12

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.image_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)