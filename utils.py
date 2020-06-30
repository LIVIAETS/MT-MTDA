import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import cmodels.mnist_net
from dotenv import load_dotenv
import os
import smtplib
import json

def get_sub_dataset_name(dataset_name, path):
    splits = path.split("/")
    if dataset_name == "Office31":
        return splits[-2]
    elif dataset_name == "ImageClef":
        return splits[-1]
    return splits[-2]

def get_config_var():
    load_dotenv()
    vars = {}
    vars["SACRED_URL"] = os.getenv("SACRED_URL")
    vars["SACRED_DB"] = os.getenv("SACRED_DB")
    vars["VISDOM_PORT"] = os.getenv("VISDOM_PORT")
    vars["SAVE_DIR"] = os.getenv("SAVE_DIR")
    vars["GMAIL_USER"] = os.getenv("GMAIL_USER")
    vars["GMAIL_PASSWORD"] = os.getenv("GMAIL_PASSWORD")
    vars["TO_EMAIL"] = os.getenv("TO_EMAIL")
    vars["SACRED_USER"] = os.getenv("SACRED_USER")
    vars["SACRED_PWD"] = os.getenv("SACRED_PWD")

    if not os.path.exists(vars["SAVE_DIR"]):
        os.makedirs(vars["SAVE_DIR"])

    return vars

all_envs =  get_config_var()
all_save_dir = all_envs["SAVE_DIR"]

def send_email(_run, result, hostname):
    if "GMAIL_PASSWORD" in all_envs:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(all_envs["GMAIL_USER"], all_envs["GMAIL_PASSWORD"])
        SUBJECT = "[EXP] Experiment {}:{} has finished with: {} Accuracy".format(_run._id, _run.experiment_info['name'],  result)
        TEXT = "Your experiment on {} has finished.\nCheers,\n".format(hostname)
        try:
            TEXT2 = json.dumps(_run.config, indent=1)
            TEXT += TEXT + TEXT2
        except:
            TEXT += "Json dumps for email did not work !"
            pass
        message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (all_envs["GMAIL_USER"], all_envs["TO_EMAIL"], SUBJECT, TEXT)
        server.sendmail(all_envs["GMAIL_USER"], all_envs["TO_EMAIL"] , message)
        server.close()

class LoggerForSacred():
    def __init__(self, visdom_logger, ex_logger=None, always_print=True):
        self.visdom_logger = visdom_logger
        self.ex_logger = ex_logger
        self.always_print = always_print


    def log_scalar(self, metrics_name, value, step):
        if self.visdom_logger is not None:
            self.visdom_logger.scalar(metrics_name, step, [value])
        if self.ex_logger is not None:
            self.ex_logger.log_scalar(metrics_name, value, step)
        if self.always_print:
            print("{}:{}/{}".format(metrics_name, value, step))

class StdOutLog():
    def __init__(self):
        self.logger = print

    def log_scalar(self, metrics_name, value, step):
        self.logger("Metrics:{}|Step:{}|Value:{}".format(metrics_name, step, value))

def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def eval(model, device, test_loader, is_debug=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if torch.cuda.device_count() > 1:
                output = model.module.nforward(data)
            else:
                output = model.nforward(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if is_debug:
                break

    acc = 100. * correct / len(test_loader.dataset)
    del output
    return acc

def eval_dan(model, device, test_loader, is_debug=False):
    model.eval()
    correct = 0
    test_loss = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_,_ = model(data, torch.FloatTensor([0]).to(device))
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target,
                                    size_average=False).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if is_debug:
                break

    acc = 100. * correct / len(test_loader.dataset)
    del output
    return 100. * correct.item() / (len(test_loader) * test_loader.batch_size)


def any_train_one_epoch(model, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if torch.cuda.device_count() > 1:
            output = model.module.nforward(data)
        else:
            output = model.nforward(data)
        output = F.log_softmax(output)
        loss = F.cross_entropy(output, target).mean()
        total_loss += float(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_break:
            break

    del loss
    del output
    # torch.cuda.empty_cache()
    return total_loss / len(train_loader)


def any_train(model, train_func, device, trainloader, testloader, optimizer, epochs, **kwargs):
    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    if "save_name" not in kwargs:
        save_name = ""
    else:
        save_name = kwargs["save_name"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    best_acc = 0
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        total_loss = train_func(model, optimizer, device, trainloader)
        acc = eval(model, device, testloader)
        if logger is not None:
            logger.log_scalar("baseline_{}_training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("baseline_{}_before_target_val_acc".format(logger_id), acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, "./{}/best_{}_{}.p".format(all_save_dir, logger_id, save_name))

    return best_acc


def main():
    train_func = any_train_one_epoch

    batch_size = 64
    test_batch_size = 64
    lr = 0.01
    momentum = 0.9
    epochs = 10

    device = torch.device("cuda")

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=1)

    model = cmodels.mnist_net.LeNet5().to(device)
    optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

    any_train(model, train_func, device, trainloader, testloader, optimizer, epochs, logger=StdOutLog(), logger_id="mnist")

if __name__ == "__main__":
    main()