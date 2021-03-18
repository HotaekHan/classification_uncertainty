# python
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pk

# pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets


# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter

# user-defined
from datagen import jsonDataset
import utils
import net_factory
from timer import Timer
from cifar_split import CIFAR_split
from temperature_scaling import ModelWithTemperature


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)

'''set random seed'''
random.seed(config['params']['seed'])
np.random.seed(config['params']['seed'])
torch.manual_seed(config['params']['seed'])
os.environ["PYTHONHASHSEED"] = str(config['params']['seed'])

'''cuda'''
if torch.cuda.is_available() and not config['gpu']['used']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

if isinstance(config['gpu']['ind'], list):
    cuda_str = 'cuda:' + str(config['gpu']['ind'][0])
elif isinstance(config['gpu']['ind'], int):
    cuda_str = 'cuda:' + str(config['gpu']['ind'])
else:
    raise ValueError('Check out gpu id in config')

device = torch.device(cuda_str if config['gpu']['used'] else "cpu")

'''Data'''
print('==> Preparing data..')
img_size = config['params']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

if config['data']['name'] == 'cifar10' or config['data']['name'] == 'cifar100':
    transform_test = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_test = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor()
    ])

def collate_fn_test(batch):
    imgs = [transform_test(x[0]) for x in batch]
    targets = [x[1] for x in batch]

    inputs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return inputs, targets

if config['data']['name'] == 'cifar100':
    num_classes = 100 - config['params']['num_exclude_class']
    train_dataset = CIFAR_split(dir_path='cifar-100-python', num_include=num_classes,
                                train=True, get_unknown=True)
    test_dataset = CIFAR_split(dir_path='cifar-100-python', num_include=num_classes,
                               train=False, get_unknown=True)
    all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
elif config['data']['name'] == 'cifar10':
    num_classes = 10 - config['params']['num_exclude_class']
    train_dataset = CIFAR_split(dir_path='cifar-10-batches-py', num_include=num_classes,
                                train=True, get_unknown=True)
    test_dataset = CIFAR_split(dir_path='cifar-10-batches-py', num_include=num_classes,
                               train=False, get_unknown=True)
    all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
else:
    raise NotImplementedError('Unsupported Dataset: ' + str(config['data']['name']))

assert all_dataset

data_loader = torch.utils.data.DataLoader(
    all_dataset, batch_size=config['params']['batch_size'],
    shuffle=False, num_workers=config['params']['workers'],
    collate_fn=collate_fn_test,
    pin_memory=True)

'''print out'''
print("num. train data : " + str(len(train_dataset)))
print("num. test data : " + str(len(test_dataset)))
print("num_classes : " + str(num_classes))

utils.print_config(config)

def view_inputs(x):
    import cv2

    x = x.detach().cpu().numpy()
    batch = x.shape[0]

    for iter_x in range(batch):
        np_x = x[iter_x]
        np_x = (np_x * 255.).astype(np.uint8)
        img = np.transpose(np_x, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('test', img)
        cv2.waitKey(0)


def draw_histogram(datalist, dir_path=None):
    if not isinstance(datalist, list):
        raise TypeError()

    data = np.array(datalist)
    np.savetxt(fname=os.path.join(dir_path, 'unknown-confidences.txt'), X=data)
    data_mean = data.mean()

    ys, xs, patches = plt.hist(data, bins=10, range=(0.0, 1.0), density=False,
                               color='b', edgecolor='black', rwidth=0.9)

    plt.xlabel('Probability')
    plt.ylabel('Num. of samples')
    title_str = f'mean of prob: {data_mean:.4f}'
    plt.title(title_str)

    if dir_path is not None:
        plt.savefig(os.path.join(dir_path, 'unknown_classes.png'))

    plt.show()


def do_test(is_scaling=False):
    ''' Model'''
    net = net_factory.load_model(config=config, num_classes=num_classes)
    net = net.to(device)
    if is_scaling is True:
        ckpt = torch.load(os.path.join(config['exp']['path'], 'model_with_temperature.pth'), map_location=device)
        weights = utils._load_weights(ckpt)
        net = ModelWithTemperature(net)
        net = net.to(device)
        missing_keys = net.load_state_dict(weights, strict=True)
        print(missing_keys)
    else:
        ckpt = torch.load(os.path.join(config['exp']['path'], 'best.pth'), map_location=device)
        weights = utils._load_weights(ckpt['net'])
        missing_keys = net.load_state_dict(weights, strict=True)
        print(missing_keys)

    '''print out net'''
    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param
    print("num. of parameters : " + str(num_parameters))


    ''' inference '''
    net.eval()

    certainties = list()
    with torch.set_grad_enabled(False):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)

            # view_inputs(inputs)
            logits = net(inputs)
            probs = logits.softmax(dim=1)

            max_probs, max_ind = probs.detach().cpu().max(dim=1)
            # max_probs = torch.ones([1], dtype=torch.float32) - max_probs
            certainties.extend(max_probs.tolist())

    draw_histogram(certainties, config['exp']['path'])


def do_test_pca():
    ''' Model'''
    net = net_factory.load_model(config=config, num_classes=num_classes, num_eigens=config['params']['num_eigens'])
    net = net.to(device)
    ckpt = torch.load(os.path.join(config['exp']['path'], 'best.pth'), map_location=device)
    weights = utils._load_weights(ckpt['net'])
    missing_keys = net.load_state_dict(weights, strict=True)
    print(missing_keys)

    '''print out net'''
    # print(net)
    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'num. of parameters: {num_parameters}')

    ''' find principle components'''
    pca_path = os.path.join(config['exp']['path'], 'pca.pkl')
    net.ipca = pk.load(open(pca_path, 'rb'))

    ''' inference '''
    net.eval()

    uncertaintes = list()
    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)

            # view_inputs(inputs)
            logits, uncertainty = net(inputs, with_uncertainty=True)
            uncertaintes.append(uncertainty)

    uncertaintes_from_first = list()
    for iter_value in uncertaintes:
        all_eigens = np.concatenate(iter_value, axis=1)
        uncertaintes_from_first.append(all_eigens.max(axis=1))

    np_uncertaintes = np.concatenate(uncertaintes_from_first, axis=0)
    draw_histogram(np_uncertaintes.tolist(), config['exp']['path'])


def apply_mc_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def do_test_beyesian():
    num_infer = config['params']['num_infer']

    ''' Model'''
    net = net_factory.load_model(config=config, num_classes=num_classes, dropout=config['params']['dropout'])
    net = net.to(device)
    ckpt = torch.load(os.path.join(config['exp']['path'], 'best.pth'), map_location=device)
    weights = utils._load_weights(ckpt['net'])
    missing_keys = net.load_state_dict(weights, strict=True)
    print(missing_keys)

    '''print out net'''
    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param
    print("num. of parameters : " + str(num_parameters))

    ''' inference '''
    net.eval()
    net.apply(apply_mc_dropout)
    print(net)

    certainties = list()
    with torch.set_grad_enabled(False):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)

            all_probs = list()
            for iter_t in range(num_infer):
                # view_inputs(inputs)
                logits = net(inputs)
                probs = logits.softmax(dim=1)
                all_probs.append(probs.detach().cpu())
            all_probs = torch.stack(all_probs)
            all_probs = all_probs.contiguous().permute(1, 2, 0)
            var, mean = torch.var_mean(all_probs, dim=2, unbiased=True)
            max_probs, max_ind = mean.max(dim=1)
            # var = var[torch.arange(0, inputs.shape[0]), max_ind]
            var = var.mean(dim=1)
            # max_probs = torch.ones([1], dtype=torch.float32) - max_probs
            certainties.extend(max_probs.tolist())

    draw_histogram(certainties, config['exp']['path'])


def do_test_ensemble():
    num_ensemble = config['params']['num_ensembles']
    nets = list()

    for iter_idx in range(num_ensemble):
        net = net_factory.load_model(config=config, num_classes=num_classes)
        net = net.to(device)

        weight_file = 'best_' + str(iter_idx) + '.pth'
        ckpt = torch.load(os.path.join(config['exp']['path'], weight_file), map_location=device)
        weights = utils._load_weights(ckpt['net'])
        missing_keys = net.load_state_dict(weights, strict=True)
        print(missing_keys)

        net.eval()
        nets.append(net)

    certainties = list()
    with torch.set_grad_enabled(False):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)

            all_probs = list()
            for net in nets:
                # view_inputs(inputs)
                logits = net(inputs)
                probs = logits.softmax(dim=1)
                all_probs.append(probs.detach().cpu())
            all_probs = torch.stack(all_probs)
            all_probs = all_probs.contiguous().permute(1, 2, 0)
            var, mean = torch.var_mean(all_probs, dim=2, unbiased=True)
            max_probs, max_ind = mean.max(dim=1)
            # var = var[torch.arange(0, inputs.shape[0]), max_ind]
            var = var.mean(dim=1)
            # max_probs = torch.ones([1], dtype=torch.float32) - max_probs
            certainties.extend(max_probs.tolist())

    draw_histogram(certainties, config['exp']['path'])


if __name__ == '__main__':
    is_bayesian = False
    is_ensemble = False
    is_scaling = False
    is_pca = False

    if 'dropout' in config['params']:
        is_bayesian = True
    elif 'num_ensembles' in config['params']:
        is_ensemble = True
    elif 'temp_scaling' in opt.config:
        is_scaling = True
    elif 'num_eigens' in config['params']:
        is_pca = True

    print('is bayesian: ' + str(is_bayesian))
    print('is ensemble: ' + str(is_ensemble))
    print('is scaling: ' + str(is_scaling))
    print('is pca: ' + str(is_pca))

    input("Press any key to continue..")

    if is_bayesian is True:
        print('select bayesian model')
        do_test_beyesian()
    elif is_ensemble is True:
        print('select ensemble model')
        do_test_ensemble()
    elif is_pca is True:
        do_test_pca()
    else:
        if is_scaling is True:
            print('select temp_scaling model')
        else:
            print('select base model')
        do_test(is_scaling)

