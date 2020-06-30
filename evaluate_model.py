import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from KD.base_kd import hinton_distillation, hinton_distillation_sw, hinton_distillation_wo_ce
import os
import DA.DA_datasets as DA_datasets
import cmodels.ResNet as ResNet
import cmodels.alexnet as AlexNet
import cmodels.DANN_GRL as DANN_GRL
import cmodels.DAN_model as DAN_model
import cmodels.LeNet as LeNet
import cmodels.BTDA_Alexnet_Office31 as BTDA_Alexnet
from utils import eval, LoggerForSacred, adjust_learning_rate, get_config_var
import DA.mmd as mmd
from utils import LoggerForSacred, send_email, get_sub_dataset_name


def main():

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
    is_debug = False

    batch_size = 16
    device = torch.device("cuda")
    student_net_func = AlexNet.alexnet
    dan_model_func_student = DANN_GRL.DANN_GRL_Alexnet
    dataset_name = "Office31"
    source_dataset_path = w
    target_dataset_paths = [a, d]
    resize_digits = 28
    is_btda = False
    finished_model_path = "best_48_webcam_and_2the_rest_kd_da_alt.p"

    source_dataloader, targets_dataloader, targets_testloader = DA_datasets.get_source_m_target_loader(dataset_name,
                                                                                                       source_dataset_path,
                                                                                                       target_dataset_paths,
                                                                                                       batch_size, 0,
                                                                                                       drop_last=True,
                                                                                                       resize=resize_digits)

    begin_pretrained = True

    if student_net_func == LeNet.LeNet:
        begin_model = dan_model_func_student(student_net_func, begin_pretrained, source_dataloader.dataset.num_classes, input_size=resize_digits).to(device)
    else:
        begin_model = dan_model_func_student(student_net_func, begin_pretrained,
                                             source_dataloader.dataset.num_classes).to(device)
    logger = LoggerForSacred(None,None, True)
    if is_btda:
        finished_model = BTDA_Alexnet.Alex_Model_Office31()
        finished_model.load_state_dict(torch.load(finished_model_path))
        finished_model = finished_model.to(device)
        finished_model.eval()
    else:
        if finished_model_path.endswith('p'):
            finished_model = torch.load(finished_model_path).to(device)
        else:
            finished_model = dan_model_func_student(student_net_func, begin_pretrained,
                                             source_dataloader.dataset.num_classes)
            finished_model.load_state_dict(torch.load(finished_model_path))
            finished_model = finished_model.to(device)
        finished_model.eval()

    s_name = get_sub_dataset_name(dataset_name, source_dataset_path)
    for i, tloader in enumerate(targets_testloader):
        acc = eval(begin_model, device, tloader, False)
        p_name = get_sub_dataset_name(dataset_name, target_dataset_paths[i])
        print("b_model_from{}_2_{}_acc:{}".format(s_name, p_name, acc))

if __name__ == "__main__":
    main()