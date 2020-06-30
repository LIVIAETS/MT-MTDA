import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from KD.base_kd import hinton_distillation, hinton_distillation_sw, hinton_distillation_wo_ce
import os
import DA.DA_datasets as DA_datasets
import cmodels.ResNet as ResNet
import cmodels.DAN_model as DAN_model
from utils import eval, LoggerForSacred, adjust_learning_rate, get_config_var

save_dir = get_config_var()["SAVE_DIR"]

def grl_multi_target_hinton_train_alt(current_ep, epochs, teacher_models, student_model, optimizer_das, optimizer_kd, device,
                         source_dataloader, targets_dataloader, T, alpha, beta, gamma, batch_norm, is_cst, is_debug=False,  **kwargs):

    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    if batch_norm:
        for teacher_model in teacher_models:
            teacher_model.train()
        student_model.train()

    total_losses = torch.zeros(len(teacher_models))
    teacher_da_temp_losses = torch.zeros(len(teacher_models))
    kd_temp_losses = torch.zeros(len(teacher_models))
    kd_target_loss = 0.
    kd_source_loss = 0.

    iter_targets = [0] * len(targets_dataloader)
    for i, d in enumerate(targets_dataloader):
        iter_targets[i] = iter(d)

    iter_source = iter(source_dataloader)

    for i in range(1, len(source_dataloader) + 1):

        data_source, label_source = iter_source.next()
        data_source = data_source.to(device)
        label_source = label_source.to(device)

        for ix, it in enumerate(iter_targets):
            try:
                data_target, _ = it.next()
            except StopIteration:
                it = iter(targets_dataloader[ix])
                data_target, _ = it.next()

            if data_target.shape[0] != data_source.shape[0]:
                data_target = data_target[: data_source.shape[0]]
            data_target = data_target.to(device)
            optimizer_das[ix].zero_grad()
            p = float(i + (current_ep -1) * len(source_dataloader)) / epochs / len(source_dataloader)
            delta = 2. / (1. + np.exp(-10 * p)) - 1
            teacher_label_source_pred, teacher_source_loss_adv = teacher_models[ix](data_source, delta)
            teacher_source_loss_cls = F.cross_entropy(F.log_softmax(teacher_label_source_pred, dim=1), label_source)

            _, teacher_target_loss_adv = teacher_models[ix](data_target, delta, source=False)
            teacher_loss_adv = teacher_source_loss_adv + teacher_target_loss_adv

            teacher_da_grl_loss = (1 - beta) * (teacher_source_loss_cls + gamma * teacher_loss_adv)
            teacher_da_temp_losses[ix] += teacher_da_grl_loss.mean().item()

            teacher_da_grl_loss.mean().backward()
            optimizer_das[ix].step() # May need to have 2 optimizers
            optimizer_das[ix].zero_grad()


            optimizer_kd.zero_grad()
            teacher_source_logits, _  = teacher_models[ix](data_source, delta, source=True)
            teacher_target_logits, _ = teacher_models[ix](data_target, delta, source=True)

            student_source_logits, _  = student_model(data_source, delta, source=True)
            student_target_logits, student_target_loss_adv = student_model(data_target, delta, source=False)

            source_kd_loss = hinton_distillation_sw(teacher_source_logits, student_source_logits, label_source, T, alpha).abs()
            if is_cst:
                target_kd_loss = hinton_distillation_wo_ce(teacher_target_logits, student_target_logits, T).abs() + alpha * student_target_loss_adv
            else:
                target_kd_loss = hinton_distillation_wo_ce(teacher_target_logits, student_target_logits, T).abs()


            kd_source_loss += source_kd_loss.mean().item()
            kd_target_loss += target_kd_loss.mean().item()

            kd_loss = beta * (target_kd_loss + source_kd_loss)
            kd_temp_losses[ix] += kd_loss.mean().item()
            total_losses[ix] += teacher_da_grl_loss.mean().item() + kd_loss.mean().item()

            kd_loss.mean().backward()
            optimizer_kd.step()
            optimizer_kd.zero_grad()

        if is_debug:
            break

    del kd_loss
    del teacher_da_grl_loss
    # torch.cuda.empty_cache()
    return total_losses / len(source_dataloader), teacher_da_temp_losses / len(source_dataloader), \
           kd_temp_losses/ len(source_dataloader)


def grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                   source_dloader, targets_dloader, targets_testloader, optimizer_das, optimizer_kd, teacher_models, student_model,
                   is_scheduler_da=True, is_scheduler_kd=False, scheduler_da=None, scheduler_kd=None, is_debug=False, save_name="", batch_norm=False, is_cst=True, **kwargs):

    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    best_student_acc = 0.
    best_teacher_acc = 0.
    epochs += 1

    for epoch in range(1, epochs):

        beta = init_beta * torch.exp(growth_rate * (epoch - 1))
        beta = beta.to(device)
        if is_scheduler_da:
            new_lr_da = init_lr_da / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            for optimizer_da in optimizer_das:
                adjust_learning_rate(optimizer_da, new_lr_da)

        if is_scheduler_kd:
            new_lr_kd = init_lr_kd / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_kd, new_lr_kd)

        total_losses, da_losses, kd_losses = grl_multi_target_hinton_train_alt(epoch, epochs, teacher_models, student_model, optimizer_das,
                                                            optimizer_kd, device, source_dloader, targets_dloader, T,
                                                            alpha, beta, gamma, batch_norm, is_cst, is_debug, logger=None)

        teachers_targets_acc = np.zeros(len(teacher_models))
        students_targets_acc = np.zeros(len(teacher_models))

        for i, d in enumerate(targets_testloader):
            teachers_targets_acc[i] = eval(teacher_models[i], device, d, is_debug)
            students_targets_acc[i] = eval(student_model, device, d, is_debug)

        total_student_target_acc = students_targets_acc.mean()
        total_teacher_target_acc = teachers_targets_acc.mean()

        if total_student_target_acc > best_student_acc:
            best_student_acc = total_student_target_acc
            torch.save({'student_model': student_model.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
                       "{}/kd_da_alt_pth_student_best_model.pth".format(save_dir))
            if save_name != "":
                torch.save(student_model, save_name)


        if logger is not None:
            logger.log_scalar("beta_epoch".format(logger_id), beta.item(), epoch)
            for i in range(len(teacher_models)):
                logger.log_scalar("training_loss_t_{}".format(i), total_losses[i].item(), epoch)
            for i in range(len(teacher_models)):
                logger.log_scalar("da_loss_t_{}".format(i), da_losses[i].item(), epoch)
            for i in range(len(teacher_models)):
                logger.log_scalar("kd_loss_{}".format(i), kd_losses[i].item(), epoch)
            logger.log_scalar("da_lr_epoch".format(logger_id), new_lr_da, epoch)
            logger.log_scalar("kd_lr_epoch".format(logger_id), optimizer_kd.param_groups[0]['lr'], epoch)
            for i in range(len(teacher_models)):
                logger.log_scalar("teacher_{}_val_target_acc".format(i, logger_id), teachers_targets_acc[i], epoch)
                logger.log_scalar("student_{}_val_target_1_acc".format(i, logger_id), students_targets_acc[i], epoch)
            logger.log_scalar("student_val_target_total_acc".format(logger_id), total_student_target_acc, epoch)
            logger.log_scalar("teacher_val_target_total_acc".format(logger_id), total_teacher_target_acc, epoch)

        if scheduler_da is not None:
            scheduler_da.step()

        if scheduler_kd is not None:
            scheduler_kd.step()

    return best_student_acc

def main():
    batch_size = 32
    test_batch_size = 32

    webcam = os.path.expanduser("~/datasets/webcam/images")
    amazon = os.path.expanduser("~/datasets/amazon/images")
    dslr = os.path.expanduser("~/datasets/dslr/images")
    is_debug = False


    epochs = 400
    init_lr_da = 0.001
    init_lr_kd = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    device = torch.device("cuda")
    T = 20
    alpha = 0.3
    init_beta = 0.1
    end_beta = 0.9

    student_pretrained = True

    if torch.cuda.device_count() > 1:
        teacher_model = nn.DataParallel(DAN_model.DANNet_ResNet(ResNet.resnet50, True)).to(device)
        student_model = nn.DataParallel(DAN_model.DANNet_ResNet(ResNet.resnet34, student_pretrained)).to(device)
    else:
        teacher_model = DAN_model.DANNet_ResNet(ResNet.resnet50, True).to(device)
        student_model = DAN_model.DANNet_ResNet(ResNet.resnet34, student_pretrained).to(device)

    growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])


    optimizer_da = torch.optim.SGD(list(teacher_model.parameters()) + list(student_model.parameters()), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)

    optimizer_kd = torch.optim.SGD(list(teacher_model.parameters()) + list(student_model.parameters()), init_lr_kd,
                                momentum=momentum, weight_decay=weight_decay)

    source_dataloader = DA_datasets.office_loader(amazon, batch_size, 1)
    target_dataloader = DA_datasets.office_loader(webcam, batch_size, 1)
    target_testloader = DA_datasets.office_test_loader(webcam, test_batch_size, 1)

    logger = LoggerForSacred(None,None, True)

    grl_multi_target_hinton_alt(init_lr_da, device, epochs, T, alpha, growth_rate, init_beta, source_dataloader,
               target_dataloader, target_testloader, optimizer_da, optimizer_kd, teacher_model, student_model, logger=logger, scheduler=None, is_debug=False)

if __name__ == "__main__":
    main()