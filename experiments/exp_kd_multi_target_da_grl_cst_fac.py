import os
from experiments.sacred_utils import custom_json_dumper, is_host_livia
from utils import get_config_var
vars = get_config_var()

from sacred import Experiment
ex = Experiment()

import kd_da_grl_alt_multi_target_cst_fac

import torch
import json
import cmodels.ResNet as ResNet
import cmodels.alexnet as AlexNet
import cmodels.LeNet as Lenet
import cmodels.DAN_model as DAN_model
import DA.DA_datasets as DA_datasets
import torch.nn as nn
import torch.nn.functional as F
import cmodels.DANN_GRL as DANN_GRL
from utils import LoggerForSacred, send_email, get_sub_dataset_name

@ex.config
def exp_config():

    #Hyper Parameters Config
    init_lr_da =  0.001
    init_lr_kd = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    device = "cuda"
    epochs = 400
    batch_size = 16
    init_beta = 0.1
    end_beta = 0.9
    T = 20
    alpha = 0.2
    gamma = 0.5
    batch_norm = True
    is_cst = True
    resize_digits = 28

    #Scheduler
    is_scheduler_da = True
    is_scheduler_kd = True
    scheduler_kd_fn = torch.optim.lr_scheduler.MultiStepLR
    scheduler_kd_steps = [250, 350]
    scheduler_kd_gamma = 0.1

    #Dataset config
    dataset_name = ""
    source_dataset_path = ""
    target_dataset_paths = []

    #Model Config
    dan_model_func = DANN_GRL.DANN_GRL_Resnet
    teacher_net_func = ResNet.resnet152
    dan_model_func_student = DAN_model.DANNet_Alexnet
    student_net_func = ResNet.resnet34
    student_pretrained = True

    #Debug config
    is_debug = False

@ex.capture()
def exp_kd_da_grl_alt(init_lr_da, init_lr_kd, momentum, weight_decay, device, epochs, batch_size, init_beta, end_beta, T, alpha, gamma, batch_norm, is_cst, resize_digits,
                      is_scheduler_da, is_scheduler_kd, scheduler_kd_fn, scheduler_kd_steps, scheduler_kd_gamma,
                      dataset_name, source_dataset_path, target_dataset_paths,
                      dan_model_func, teacher_net_func, dan_model_func_student, student_net_func, student_pretrained,
                      is_debug, _run):

    source_dataloader, targets_dataloader, targets_testloader = DA_datasets.get_source_m_target_loader(dataset_name,
                                                                                                   source_dataset_path,
                                                                                                   target_dataset_paths,
                                                                                                   batch_size, 0, drop_last=True, resize=resize_digits)
    teacher_models = []
    for p in target_dataset_paths:
        teacher_models.append(dan_model_func(teacher_net_func, True, source_dataloader.dataset.num_classes).to(device))

    if student_net_func == Lenet.LeNet or student_net_func == Lenet.MTDA_ITA_classifier:
        student_model = dan_model_func_student(student_net_func, student_pretrained, source_dataloader.dataset.num_classes, input_size=resize_digits).to(device)
    else:
        student_model = dan_model_func_student(student_net_func, student_pretrained,
                                               source_dataloader.dataset.num_classes).to(device)

    if torch.cuda.device_count() > 1:
        for i, tm in enumerate(teacher_models):
            teacher_models[i] = nn.DataParallel(tm).to(device)
        student_model = nn.DataParallel(student_model).to(device)

    logger = LoggerForSacred(None, ex)

    growth_rate = torch.zeros(1)
    if init_beta != 0.0:
        growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])

    optimizer_das = []
    for tm in teacher_models:
        optimizer_das.append(torch.optim.SGD(tm.parameters(), init_lr_da,
                                    momentum=momentum, weight_decay=weight_decay))

    optimizer_kd = torch.optim.SGD(student_model.parameters(), init_lr_kd,
                                momentum=momentum, weight_decay=weight_decay)

    scheduler_kd = None
    if scheduler_kd_fn is not None:
        scheduler_kd = scheduler_kd_fn(optimizer_kd, scheduler_kd_steps, scheduler_kd_gamma)

    if dataset_name != "Digits" and dataset_name != "Digits_no_split":
        source_name_1 = get_sub_dataset_name(dataset_name, source_dataset_path)
    else:
        source_name_1 = source_dataset_path

    save_name = "best_{}_{}_and_2{}_kd_da_alt.p".format(_run._id, source_name_1, "the_rest")

    best_student_acc = kd_da_grl_alt_multi_target_cst_fac.grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                                                source_dataloader, targets_dataloader, targets_testloader, optimizer_das, optimizer_kd,
                                                teacher_models, student_model,
                                                logger=logger,
                                                is_scheduler_da=is_scheduler_da, is_scheduler_kd=is_scheduler_kd, scheduler_kd=None, scheduler_da=None,
                                                is_debug=is_debug, run=_run, save_name=save_name, batch_norm=batch_norm, is_cst=is_cst)


    conf_path = "{}/{}_{}.json".format("all_confs", _run._id, best_student_acc)
    with open(conf_path, 'w') as cf:
        json.dump(_run.config, cf, default=custom_json_dumper)

    #send_email(_run, best_student_acc, os.uname()[1])

    return best_student_acc

@ex.main
def run_exp():
    return exp_kd_da_grl_alt()

if __name__ == "__main__":

    if os.name != 'nt':
        a = os.path.expanduser('~/datasets/amazon/images')
        w = os.path.expanduser('~/datasets/webcam/images')
        d = os.path.expanduser('~/datasets/dslr/images')

        Ar = os.path.expanduser('~/datasets/OfficeHome/Art')
        Cl = os.path.expanduser('~/datasets/OfficeHome/Clipart')
        Pr = os.path.expanduser('~/datasets/OfficeHome/Product')
        Rw = os.path.expanduser('~/datasets/OfficeHome/RealWorld')

        i = os.path.expanduser('~/datasets/image-clef/i')
        p = os.path.expanduser('~/datasets/image-clef/p')
        c = os.path.expanduser('~/datasets/image-clef/c')

        p_ar = os.path.expanduser('~/datasets/pacs/art_painting')
        p_c = os.path.expanduser('~/datasets/pacs/cartoon')
        p_p = os.path.expanduser('~/datasets/pacs/photo')
        p_s = os.path.expanduser('~/datasets/pacs/sketch')

    mt = "MNIST"
    mm = "MNIST-M"
    sv = "SVHN"
    up = "USPS"
    sy = "SY"

    ex.run(config_updates={'dataset_name': 'OfficeHome',
                           'source_dataset_path': Pr,
                           'target_dataset_paths': [Rw, Ar, Cl],
                           'init_beta': 0.1,
                           'end_beta': 0.5,
                           'init_lr_da': 0.001,
                           'init_lr_kd': 0.01,
                           'T': 20,
                           'alpha': 0.5,
                           'epochs': 200,
                           'scheduler_kd_fn': None,
                           'batch_norm': True,
                           'batch_size': 32,
                           'teacher_net_func': ResNet.resnet50,
                           'dan_model_func_student': DANN_GRL.DANN_GRL_Alexnet,
                           'student_net_func': AlexNet.alexnet},
           options={"--name": 'Search_cst_Pr_2_Ar_Rw_Cl_rerun_kd_da_alt_resnet50-alexnet'})
    ex.run(config_updates={'dataset_name': 'OfficeHome',
                           'source_dataset_path': Rw,
                           'target_dataset_paths': [Pr, Ar, Cl],
                           'init_beta': 0.1,
                           'end_beta': 0.5,
                           'init_lr_da': 0.001,
                           'init_lr_kd': 0.01,
                           'T': 20,
                           'alpha': 0.5,
                           'epochs': 200,
                           'scheduler_kd_fn': None,
                           'batch_norm': True,
                           'batch_size': 8,
                           'teacher_net_func': ResNet.resnet50,
                           'dan_model_func_student': DANN_GRL.DANN_GRL_Alexnet,
                           'student_net_func': AlexNet.alexnet},
           options={"--name": 'Search_cst_Rw_2_Ar_Pr_Cl_rerun_kd_da_alt_resnet50-alexnet'})
    ex.run(config_updates={'dataset_name': 'OfficeHome',
                           'source_dataset_path': Ar,
                           'target_dataset_paths': [Rw, Cl, Pr],
                           'init_beta': 0.1,
                           'end_beta': 0.5,
                           'init_lr_da': 0.001,
                           'init_lr_kd': 0.01,
                           'T': 20,
                           'alpha': 0.5,
                           'epochs': 200,
                           'scheduler_kd_fn': None,
                           'batch_norm': True,
                           'batch_size': 32,
                           'teacher_net_func': ResNet.resnet50,
                           'dan_model_func_student': DANN_GRL.DANN_GRL_Alexnet,
                           'student_net_func': AlexNet.alexnet},
           options={"--name": 'Search_cst_Ar_2_Rw_Cl_Pr_rerun_kd_da_alt_resnet50-alexnet'})
    ex.run(config_updates={'dataset_name': 'OfficeHome',
                           'source_dataset_path': Cl,
                           'target_dataset_paths': [Pr, Rw, Ar],
                           'init_beta': 0.1,
                           'end_beta': 0.5,
                           'init_lr_da': 0.001,
                           'init_lr_kd': 0.01,
                           'T': 20,
                           'alpha': 0.5,
                           'epochs': 200,
                           'scheduler_kd_fn': None,
                           'batch_norm': True,
                           'batch_size': 32,
                           'teacher_net_func': ResNet.resnet50,
                           'dan_model_func_student': DANN_GRL.DANN_GRL_Alexnet,
                           'student_net_func': AlexNet.alexnet},
           options={"--name": 'Search_cst_Cl_2_Rw_Pr_Ar_rerun_kd_da_alt_resnet50-alexnet'})
