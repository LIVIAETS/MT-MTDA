import math
import torch.nn.functional as F
import numpy as np
import torch

def train_dan(epoch, epochs, model_dan, optimizer, device, train_loader_source, train_loader_target, is_debug=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model_dan.train()

    iter_source = iter(train_loader_source)
    iter_target = iter(train_loader_target)

    for i in range(1, len(train_loader_source)):

        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()

        if data_target.shape[0] < train_loader_source.batch_size:
            iter_target = iter(train_loader_target)
            data_target, _ = iter_target.next()

        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target = data_target.to(device)

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model_dan(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + np.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + gamma * loss_mmd
        total_loss += float(loss.item())
        loss.backward()
        optimizer.step()

        if is_debug:
            break

    del loss_cls
    del loss_mmd
    del loss
    torch.cuda.empty_cache()
    return total_loss / (len(train_loader_source) * train_loader_source.batch_size)

def eval_dan(model_dan, device, test_loader_target, is_debug=False):
    model_dan.eval()
    test_loss = 0
    correct = 0.
    for data, target in test_loader_target:
        data, target = data.to(device), target.to(device)
        s_output, t_output = model_dan(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).item()  # sum up batch loss
        pred = s_output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if is_debug:
            return 100. * correct.item() / (len(test_loader_target) * test_loader_target.batch_size)

    del test_loss
    torch.cuda.empty_cache()
    return 100. * correct.item() / (len(test_loader_target) * test_loader_target.batch_size)