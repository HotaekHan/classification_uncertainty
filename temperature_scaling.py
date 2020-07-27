# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch
from torch import nn, optim
import torchvision.transforms as transforms

# user-defined
import utils
from cifar_split import CIFAR_split
import net_factory
from ece_loss import ECELoss

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.model = self.model.to(device)
        # nll_criterion = nn.CrossEntropyLoss().cuda()
        # ece_criterion = _ECELoss().cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = list()
        labels_list = list()
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


if __name__ == '__main__':
    config = utils.get_config(opt.config)

    '''make output folder'''
    if not os.path.exists(config['exp']['path']):
        os.makedirs(config['exp']['path'], exist_ok=False)

    if not os.path.exists(os.path.join(config['exp']['path'], 'config.yaml')):
        shutil.copy(opt.config, os.path.join(config['exp']['path'], 'config.yaml'))
    else:
        os.remove(os.path.join(config['exp']['path'], 'config.yaml'))
        shutil.copy(opt.config, os.path.join(config['exp']['path'], 'config.yaml'))

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

    '''set random seed'''
    random.seed(config['params']['seed'])
    np.random.seed(config['params']['seed'])
    torch.manual_seed(config['params']['seed'])
    os.environ["PYTHONHASHSEED"] = str(config['params']['seed'])

    ''' data load'''
    print('==> Preparing data..')
    img_size = config['params']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1]))

    transform_test = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def collate_fn_test(batch):
        imgs = [transform_test(x[0]) for x in batch]
        targets = [x[1] for x in batch]

        inputs = torch.stack(imgs)
        targets = torch.tensor(targets)

        return inputs, targets

    if config['data']['name'] == 'cifar100':
        num_classes = 100 - config['params']['num_exclude_class']
        # train_data = datasets.CIFAR100(os.getcwd(), train=True, download=True, transform=None)
        train_data = CIFAR_split(dir_path='cifar-100-python', num_exclude=config['params']['num_exclude_class'],
                                 train=True)
        num_train = len(train_data)
        num_valid = int(num_train * 0.2)
        num_train = num_train - num_valid

        train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [num_train, num_valid])
    elif config['data']['name'] == 'cifar10':
        num_classes = 10 - config['params']['num_exclude_class']
        # train_data = datasets.CIFAR10(os.getcwd(), train=True, download=True, transform=None)
        if config['params']['num_exclude_class'] > 8:
            raise ValueError('cifar10 has 10 classes. the number of exclude classes is over than num. of classes')

        train_data = CIFAR_split(dir_path='cifar-10-batches-py', num_exclude=config['params']['num_exclude_class'],
                                 train=True)
        num_train = len(train_data)
        num_valid = int(num_train * 0.2)
        num_train = num_train - num_valid

        train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [num_train, num_valid])
    else:
        raise NotImplementedError('Unsupported Dataset: ' + str(config['data']['name']))

    assert valid_dataset

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config['params']['batch_size'],
        shuffle=False, num_workers=config['params']['workers'],
        collate_fn=collate_fn_test,
        pin_memory=True)

    ''' Load Model'''
    net = net_factory.load_model(config=config, num_classes=num_classes)
    net = net.to(device)
    ckpt = torch.load(config['basenet']['path'], map_location=device)
    weights = utils._load_weights(ckpt['net'])
    missing_keys = net.load_state_dict(weights, strict=False)
    print(missing_keys)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    temp_model = ModelWithTemperature(net)
    temp_model = temp_model.to(device)

    # Tune the model temperature, and save the results
    temp_model.set_temperature(valid_loader, device)
    model_filename = os.path.join(config['exp']['path'], 'model_with_temperature.pth')
    torch.save(temp_model.state_dict(), model_filename)
    print('Temperature scaled model save to %s' % model_filename)
    print('Done!')