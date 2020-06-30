import numpy as np
from utils import adjust_learning_rate, eval, LoggerForSacred
import torch.nn.functional as F
import os
import torch
import DA.DA_datasets as DA_datasets
from visdom_logger.logger import VisdomLogger
import cmodels.DAN_model as DAN_model
import torchvision.models as models

torch.cuda.manual_seed(8)

def dann_grl_one_epoch(current_ep, epochs, teacher_model, optimizer, device, source_dataloader, target_dataloader, is_debug=False, **kwargs):

    teacher_da_grl_temp_loss = 0.

    iter_source = iter(source_dataloader)
    iter_target = iter(target_dataloader)
    teacher_model.train()

    for i in range(1, len(source_dataloader) + 1):

        data_source, label_source = iter_source.next()
        try:
            data_target, _ = iter_target.next()
        except StopIteration:
            iter_target = iter(target_dataloader)

        if data_source.shape[0] != data_target.shape[0]:
            if data_target.shape[0] < source_dataloader.batch_size:
                iter_target = iter(target_dataloader)
                data_target, _ = iter_target.next()

            if data_source.shape[0] < source_dataloader.batch_size:
                data_target = data_target[:data_source.shape[0]]


        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target = data_target.to(device)
        optimizer.zero_grad()

        p = float(i + (current_ep - 1) * len(source_dataloader)) / epochs / len(source_dataloader)
        delta = 2. / (1. + np.exp(-10 * p)) - 1
        teacher_label_source_pred, teacher_source_loss_adv = teacher_model(data_source, delta)
        teacher_source_loss_cls = F.cross_entropy(F.log_softmax(teacher_label_source_pred, dim=1), label_source)

        _, teacher_target_loss_adv = teacher_model(data_target, delta, source=False)
        teacher_loss_adv = teacher_source_loss_adv + teacher_target_loss_adv

        teacher_da_grl_loss = 1. * (teacher_source_loss_cls + 0.5 * teacher_loss_adv)
        teacher_da_grl_temp_loss += teacher_da_grl_loss.mean().item()
        teacher_da_grl_loss.mean().backward()

        optimizer.step()

        if is_debug:
            break

    return teacher_da_grl_temp_loss / len(source_dataloader)

def dann_grl_train(epochs, lr, model_dan, train_loader_source, device, train_loader_target, testloader_target, optimizer, logger=None,
                   logger_id="", scheduler=None, is_debug=False):

    epochs += 1
    best_acc = 0.
    for epoch in range(1, epochs):
        total_loss = dann_grl_one_epoch(epoch, epochs, model_dan, optimizer, device, train_loader_source, train_loader_target, is_debug)
        acc = eval(model_dan, device, testloader_target, is_debug=is_debug)

        if scheduler is not None:
            scheduler.step()
        else:
            new_lr = lr / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer, new_lr)

        if acc > best_acc:
            best_acc = acc

        if logger is not None:
            logger.log_scalar("da_lr_epoch".format(logger_id), new_lr, epoch)
            logger.log_scalar("baseline_{}_training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("baseline_{}_target_val_acc".format(logger_id), acc, epoch)

    return model_dan, optimizer, best_acc


if __name__ == "__main__":
    batch_size = 32
    test_batch_size = 32

    #train_path = "/home/ens/AN88740/dataset/webcam/images"
    #test_path = "/home/ens/AN88740/dataset/amazon/images"

    webcam = os.path.expanduser("~/datasets/webcam/images")
    amazon = os.path.expanduser("~/datasets/amazon/images")
    dslr = os.path.expanduser("~/datasets/dslr/images")

    epochs = 200
    lr = 0.01
    device = torch.device("cuda")

    train_loader_source = DA_datasets.office_loader(webcam, batch_size, 0)
    train_loader_target = DA_datasets.office_loader(amazon, batch_size, 0)
    testloader_1_target = DA_datasets.office_test_loader(amazon, test_batch_size, 0)

    logger = VisdomLogger(port=9000)
    logger = LoggerForSacred(logger)

    #model_dan = DAN_model.DANNet_ResNet(ResNet.resnet50, True).to(device)
    model_dan = DAN_model.DANNetVGG16(models.vgg16, True).to(device)

    optimizer = torch.optim.SGD(model_dan.parameters(), momentum=0.9, lr=lr, weight_decay=5e-4)
    dann_grl_train(epochs, lr, model_dan, train_loader_source, device, train_loader_target, testloader_1_target, optimizer, logger=logger,
                   logger_id="", scheduler=None, is_debug=False)