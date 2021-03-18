# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter

# user-defined
from datagen import jsonDataset
from landmark_dataset import Landmark_dataset
from optimizer import scheduled_optim
import mixup
import label_smoothing
import utils
import net_factory
from cifar_split import CIFAR_split


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)
start_epoch = 0  # start from epoch 0 or last epoch

'''make output folder'''
if not os.path.exists(config['exp']['path']):
    os.makedirs(config['exp']['path'], exist_ok=False)

if not os.path.exists(os.path.join(config['exp']['path'], 'config.yaml')):
    shutil.copy(opt.config, os.path.join(config['exp']['path'], 'config.yaml'))
else:
    os.remove(os.path.join(config['exp']['path'], 'config.yaml'))
    shutil.copy(opt.config, os.path.join(config['exp']['path'], 'config.yaml'))

'''set random seed'''
random.seed(config['params']['seed'])
np.random.seed(config['params']['seed'])
torch.manual_seed(config['params']['seed'])
os.environ["PYTHONHASHSEED"] = str(config['params']['seed'])

'''variables'''
best_valid_loss = float('inf')
global_iter_train = 0
global_iter_valid = 0

'''cuda'''
if torch.cuda.is_available() and not config['gpu']['used']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

is_data_parallel = False
if isinstance(config['gpu']['ind'], list):
    is_data_parallel = True
    cuda_str = 'cuda:' + str(config['gpu']['ind'][0])
elif isinstance(config['gpu']['ind'], int):
    cuda_str = 'cuda:' + str(config['gpu']['ind'])
else:
    raise ValueError('Check out gpu id in config')

device = torch.device(cuda_str if config['gpu']['used'] else "cpu")

'''tensorboard'''
summary_writer = SummaryWriter(os.path.join(config['exp']['path'], 'log'))

'''Data'''
print('==> Preparing data..')
img_size = config['params']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

transform_train = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.RandomCrop(size=img_size, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def collate_fn_train(batch):
    imgs = [transform_train(x[0]) for x in batch]
    targets = [x[1] for x in batch]

    inputs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return inputs, targets

def collate_fn_test(batch):
    imgs = [transform_test(x[0]) for x in batch]
    targets = [x[1] for x in batch]

    inputs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return inputs, targets


if config['data']['name'] == 'cifar100':
    num_classes = 100 - config['params']['num_exclude_class']
    # train_data = datasets.CIFAR100(os.getcwd(), train=True, download=True, transform=None)
    if config['params']['num_exclude_class'] > 99:
        raise ValueError('cifar10 has 10 classes. the number of exclude classes is over than num. of classes')

    train_data = CIFAR_split(dir_path='cifar-100-python', num_include=num_classes,
                             train=True)
    num_train = len(train_data)
    num_valid = int(num_train * 0.2)
    num_train = num_train - num_valid

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=train_data,
                                                                 lengths=[num_train, num_valid],
                                                                 generator=torch.Generator().manual_seed(
                                                                     config['params']['seed']))
elif config['data']['name'] == 'cifar10':
    num_classes = 10 - config['params']['num_exclude_class']
    # train_data = datasets.CIFAR10(os.getcwd(), train=True, download=True, transform=None)
    if config['params']['num_exclude_class'] > 9:
        raise ValueError('cifar10 has 10 classes. the number of exclude classes is over than num. of classes')

    train_data = CIFAR_split(dir_path='cifar-10-batches-py', num_include=num_classes,
                             train=True)
    num_train = len(train_data)
    num_valid = int(num_train * 0.2)
    num_train = num_train - num_valid

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=train_data,
                                                                 lengths=[num_train, num_valid],
                                                                 generator=torch.Generator().manual_seed(
                                                                     config['params']['seed']))
elif config['data']['name'] == 'its':
    target_classes = config['params']['classes'].split('|')
    num_classes = len(target_classes)
    train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes)

    valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes)
elif config['data']['name'] == 'landmark':
    train_data = Landmark_dataset(root='/data/kaggle/dacon_landmark_korea/public',
                                  is_train=True)
    num_classes = train_data.num_classes
    num_data = len(train_data)
    num_train = int(num_data * 0.7)
    num_valid = num_data - num_train
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=train_data,
                                                                 lengths=[num_train, num_valid],
                                                                 generator=torch.Generator().manual_seed(
                                                                     config['params']['seed']))
else:
    raise NotImplementedError('Unsupported Dataset: ' + str(config['data']['name']))

assert train_dataset
assert valid_dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['params']['batch_size'],
    shuffle=True, num_workers=config['params']['workers'],
    collate_fn=collate_fn_train,
    pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config['params']['batch_size'],
    shuffle=False, num_workers=config['params']['workers'],
    collate_fn=collate_fn_test,
    pin_memory=True)

dataloaders = {'train': train_loader, 'valid': valid_loader}

''' Model'''
if 'dropout' in config['params']:
    net = net_factory.load_model(config=config, num_classes=num_classes, dropout=config['params']['dropout'])
elif 'num_eigens' in config['params']:
    net = net_factory.load_model(config=config, num_classes=num_classes, num_eigens=config['params']['num_eigens'])
else:
    net = net_factory.load_model(config=config, num_classes=num_classes, dropout=None)
net = net.to(device)

'''print out net'''
print(net)
num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("num. of parameters : " + str(num_parameters))

'''set data parallel'''
if is_data_parallel is True:
    net = torch.nn.DataParallel(module=net, device_ids=config['gpu']['ind'])

'''loss'''
# criterion = nn.CrossEntropyLoss(reduction='mean')
criterion = nn.KLDivLoss(reduction='batchmean')

'''optimizer'''
optim = scheduled_optim(params=filter(lambda p: p.requires_grad, net.parameters()), config=config)
optimizer = optim.construct_optimizer()


'''set pre-trained'''
if config['model']['pretrained'] != 'None':
    print('loading pretrained model from %s' % config['model']['pretrained'])
    ckpt = torch.load(config['model']['pretrained'], map_location=device)
    weights = utils._load_weights(ckpt['net'])
    missing_keys = net.load_state_dict(weights, strict=False)
    print(missing_keys)
    start_epoch = ckpt['epoch'] + 1
    if config['model']['is_finetune'] is False:
        best_valid_loss = ckpt['loss']
        global_iter_train = ckpt['global_train_iter']
        global_iter_valid = ckpt['global_valid_iter']
    else:
        start_epoch = 0
    optimizer.load_state_dict(state_dict=ckpt['optimizer'])
    # scheduler_for_lr = ckpt['scheduler']

'''print out'''
print(optimizer)
print("Size of batch : " + str(train_loader.batch_size))
print("transform : " + str(transform_train))
print("num. train data : " + str(len(train_dataset)))
print("num. valid data : " + str(len(valid_dataset)))
print("num_classes : " + str(num_classes))

utils.print_config(config)

input("Press any key to continue..")

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


def iterate(epoch, phase):
    is_train = True
    if phase == 'train':
        is_train = True
    elif phase == 'valid':
        is_train = False
    else:
        raise ValueError('Unrecognized phase: ' + str(phase))

    if is_train is True:
        net.train()
        '''learning rate scheduling'''
        if config['optimizer']['use_adam'] is False:
            lr = optim.get_epoch_lr(epoch)
            optim.set_lr(optimizer, lr)
    else:
        net.eval()

    phase_dataloader = dataloaders[phase]

    acc_loss = 0.
    is_saved = False

    global best_valid_loss
    global global_iter_valid
    global global_iter_train

    with torch.set_grad_enabled(is_train):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(phase_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # view_inputs(inputs)
            if is_train is True:
                '''mix up'''
                inputs, targets_a, targets_b, lam = mixup.mixup_data(inputs, targets,
                                                                     device, float(config['params']['mixup_alpha']))
                # inputs, targets_a, targets_b = map(Variable, (inputs,
                #                                               targets_a, targets_b))

                '''label smoothing'''
                targets_a = label_smoothing.smooth_one_hot(true_labels=targets_a, classes=num_classes,
                                                           smoothing=float(config['params']['label_smoothing']))
                targets_b = label_smoothing.smooth_one_hot(true_labels=targets_b, classes=num_classes,
                                                           smoothing=float(config['params']['label_smoothing']))
            else:
                targets = label_smoothing.smooth_one_hot(true_labels=targets, classes=num_classes,
                                                         smoothing=0.0)

            # view_inputs(inputs)
            if config['model']['type'] == 'arcface':
                if is_train is True:
                    logits = net(inputs, targets_a)
                else:
                    logits = net(inputs, targets)
            else:
                logits = net(inputs)
            outputs = logits.log_softmax(dim=1)

            if is_train is True:
                loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1, keepdim=True)

            if is_train is True:
                targets_a = targets_a.argmax(dim=1, keepdim=True)
                targets_b = targets_b.argmax(dim=1, keepdim=True)
                accuracy = (lam * preds.eq(targets_a).float().sum()
                            + (1 - lam) * preds.eq(targets_b).float().sum())
                accuracy = accuracy / (targets_a.shape[0] + targets_b.shape[0])
            else:
                targets = targets.argmax(dim=1, keepdim=True)
                accuracy = preds.eq(targets).float().mean()

            acc_loss += loss.item()
            avg_loss = acc_loss / (batch_idx + 1)
            print('[%s] epoch: %3d | iter: %4d | loss: %.3f | avg_loss: %.3f | accuracy: %.3f'
                  % (phase, epoch, batch_idx, loss.item(), avg_loss, accuracy))

            if is_train is True:
                summary_writer.add_scalar('train/loss', loss.item(), global_iter_train)
                summary_writer.add_scalar('train/acc', accuracy, global_iter_train)
                global_iter_train += 1
            else:
                summary_writer.add_scalar('valid/loss', loss.item(), global_iter_valid)
                summary_writer.add_scalar('valid/acc', accuracy, global_iter_valid)
                global_iter_valid += 1

        state = {
            'net': net.state_dict(),
            'loss': best_valid_loss,
            'epoch': epoch,
            'lr': config['optimizer']['lr'],
            'batch': config['params']['batch_size'],
            'global_train_iter': global_iter_train,
            'global_valid_iter': global_iter_valid,
            'optimizer': optimizer.state_dict()
        }

        if is_train is True:
            print('[Train] Saving..')
            # torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')
            torch.save(state, os.path.join(config['exp']['path'], 'latest.pth'))
        else:
            # check whether better model or not
            if avg_loss < best_valid_loss:
                best_valid_loss = avg_loss
                is_saved = True

            if is_saved is True:
                print('[Valid] Saving..')
                # torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')
                torch.save(state, os.path.join(config['exp']['path'], 'best.pth'))


if __name__ == '__main__':
    for epoch in range(start_epoch, config['params']['epoch'], 1):
        iterate(epoch=epoch, phase='train')
        iterate(epoch=epoch, phase='valid')
    summary_writer.close()

    print("best valid loss : " + str(best_valid_loss))