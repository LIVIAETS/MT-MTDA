import math
import torch.nn.functional as F
import numpy as np
import torch

def train_dann(epoch, epochs, model_dann, optimizer, device, train_loader_source, train_loader_target, is_debug=False):
    total_loss = 0.
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    # One epoch step gradient for target
    optimizer.zero_grad()
    model_dann.train()

    iter_source = iter(train_loader_source)
    iter_target = iter(train_loader_target)
    len_dataloader = min(len(train_loader_source), len(train_loader_target))

    for i in range(1, len(train_loader_source)):

        optimizer.zero_grad()
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1


        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.to(device), label_source.to(device)
        s_domain_label = torch.zeros(train_loader_source.batch_size).long().to(device)
        class_output, domain_output = model_dann(data_source, alpha)
        err_s_label = loss_class(class_output, label_source)
        err_s_domain = loss_domain(domain_output, s_domain_label)


        data_target, _ = iter_target.next()
        if data_target.shape[0] < train_loader_source.batch_size:
            iter_target = iter(train_loader_target)
            data_target, _ = iter_target.next()
        data_target = data_target.to(device)
        t_domain_label = torch.ones(train_loader_source.batch_size).long().to(device)
        _, domain_output = model_dann(data_target, alpha)
        err_t_domain = loss_domain(domain_output, t_domain_label)
        err = err_s_label + err_t_domain + err_s_domain
        err.backward()
        total_loss += float(err.item())
        optimizer.step()

        if is_debug:
            break

    del err
    # torch.cuda.empty_cache()
    return total_loss / (len(train_loader_source) * train_loader_source.batch_size)

def eval_dann(model_dann, device, test_loader_target, is_debug=False):
    model_dann.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader_target:
        data, target = data.to(device), target.to(device)
        class_output, _ = model_dann(data, data)
        test_loss += F.nll_loss(F.log_softmax(class_output, dim=1), target, size_average=False).item()  # sum up batch loss
        pred = class_output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if is_debug:
            return correct
    test_loss /= len(test_loader_target)
    return 100. * correct / (len(test_loader_target) * test_loader_target.batch_size)