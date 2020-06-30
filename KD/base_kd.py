import torch.nn.functional as F
import torch
import torch.nn as nn

def temp_softmax(z, T, dim=1):
    return F.softmax(z/T, dim=dim)

def temp_log_softmax(z, T, dim=1):
    return F.log_softmax(z / T, dim=dim)

def hinton_distillation(y_teacher, y_student, labels, T, alpha, dist_loss=F.kl_div):
    p = F.log_softmax(y_teacher/T, dim=1)
    q = F.softmax(y_student/T, dim=1)
    if dist_loss == F.kl_div:
        l_kl = dist_loss(p, q, size_average=False) * (T**2) / y_teacher.shape[0]
    elif dist_loss == F.mse_loss:
        l_kl = dist_loss(p, q, size_average=False) / y_teacher.shape[0]
    l_ce = F.cross_entropy(y_teacher, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def hinton_distillation_sw(y_teacher, y_student, labels, T, alpha, dist_loss=F.kl_div):
    p = F.log_softmax(y_teacher/T, dim=1)
    q = F.softmax(y_student/T, dim=1)
    if dist_loss == F.kl_div:
        l_kl = dist_loss(p, q, size_average=False) * (T**2) / y_teacher.shape[0]
    elif dist_loss == F.mse_loss:
        l_kl = dist_loss(p, q, size_average=False) / y_teacher.shape[0]
    l_ce = F.cross_entropy(y_student, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def hinton_distillation_wo_ce(y_teacher, y_student, T, dist_loss=F.kl_div):
    p = F.log_softmax(y_teacher/T, dim=1)
    q = F.softmax(y_student/T, dim=1)
    if dist_loss == F.kl_div:
        l_kl = dist_loss(p, q, size_average=False) * (T**2) / y_teacher.shape[0]
    elif dist_loss == F.mse_loss:
        l_kl = dist_loss(p, q, size_average=False) / y_teacher.shape[0]
    return l_kl # To be multipled to alpha

class KD_Fusion(nn.Module):
    def __init__(self, num_teachers, num_classes):
        super(KD_Fusion, self).__init__()
        self.fc_fused_1 = nn.Linear(num_classes * num_teachers, 4096)
        self.relu_1 = nn.ReLU()
        self.fc_fused_2 = nn.Linear(4096, num_classes)
    def forward(self, logits):
        concat_logits = torch.cat(logits, dim=1)
        out = self.fc_fused_1(concat_logits)
        out = self.relu_1(out)
        out = self.fc_fused_2(out)
        return out

class KD_Discriminator(nn.Module):
    def __init__(self, num_teachers, num_classes):
        super(KD_Fusion, self).__init__()
        self.fc_d_1 = nn.Linear(num_classes, 4096)
        self.relu_1 = nn.ReLU()
        self.fc_d_2 = nn.Linear(4096, num_teachers)

    def forward(self, logits):
        concat_logits = torch.cat(logits, dim=1)
        out = self.fc_d_1(concat_logits)
        out = self.relu_1(out)
        out = self.fc_d_2(out)
        return out