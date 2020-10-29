from models import ResNet, ResNetD, ShuffleNetV2, TResNet, Mobilenetv3, EfficientNet, RegNet, ResNest, ReXNet, Arcface


def load_model(config, num_classes, dropout=None):
    if config['model']['type'] == 'resnet':
        if config['model']['arch'] == 'resnet18':
            net = ResNet.resnet18(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout)
        elif config['model']['arch'] == 'resnet50':
            net = ResNet.resnet50(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout)
        elif config['model']['arch'] == 'resnext50':
            net = ResNet.resnext50_32x4d(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout)
        elif config['model']['arch'] == 'resnet50d':
            net = ResNetD.resnet50d(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'tresnet':
        if config['model']['arch'] == 'tresnetm':
            net = TResNet.TResnetM(num_classes=num_classes)
        elif config['model']['arch'] == 'tresnetl':
            net = TResNet.TResnetL(num_classes=num_classes)
        elif config['model']['arch'] == 'tresnetxl':
            net = TResNet.TResnetXL(num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'regnet':
        regnet_config = dict()
        if config['model']['arch'] == 'regnetx-200mf':
            regnet_config['depth'] = 13
            regnet_config['w0'] = 24
            regnet_config['wa'] = 36.44
            regnet_config['wm'] = 2.49
            regnet_config['group_w'] = 8
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnetx-600mf':
            regnet_config['depth'] = 16
            regnet_config['w0'] = 48
            regnet_config['wa'] = 36.97
            regnet_config['wm'] = 2.24
            regnet_config['group_w'] = 24
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnetx-4.0gf':
            regnet_config['depth'] = 23
            regnet_config['w0'] = 96
            regnet_config['wa'] = 38.65
            regnet_config['wm'] = 2.43
            regnet_config['group_w'] = 40
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnetx-6.4gf':
            regnet_config['depth'] = 17
            regnet_config['w0'] = 184
            regnet_config['wa'] = 60.83
            regnet_config['wm'] = 2.07
            regnet_config['group_w'] = 56
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnety-200mf':
            regnet_config['depth'] = 13
            regnet_config['w0'] = 24
            regnet_config['wa'] = 36.44
            regnet_config['wm'] = 2.49
            regnet_config['group_w'] = 8
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnety-600mf':
            regnet_config['depth'] = 15
            regnet_config['w0'] = 48
            regnet_config['wa'] = 32.54
            regnet_config['wm'] = 2.32
            regnet_config['group_w'] = 16
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnety-4.0gf':
            regnet_config['depth'] = 22
            regnet_config['w0'] = 96
            regnet_config['wa'] = 31.41
            regnet_config['wm'] = 2.24
            regnet_config['group_w'] = 64
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['model']['arch'] == 'regnety-6.4gf':
            regnet_config['depth'] = 25
            regnet_config['w0'] = 112
            regnet_config['wa'] = 33.22
            regnet_config['wm'] = 2.27
            regnet_config['group_w'] = 72
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'resnest':
        if config['model']['arch'] == 'resnest50':
            net = ResNest.resnest50(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'resnest101':
            net = ResNest.resnest101(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'efficient':
        if config['model']['arch'] == 'b0':
            net = EfficientNet.efficientnet_b0(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b1':
            net = EfficientNet.efficientnet_b1(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b2':
            net = EfficientNet.efficientnet_b2(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b3':
            net = EfficientNet.efficientnet_b3(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b4':
            net = EfficientNet.efficientnet_b4(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b5':
            net = EfficientNet.efficientnet_b5(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b6':
            net = EfficientNet.efficientnet_b6(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'assembled':
        raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
        pass
    elif config['model']['type'] == 'shufflenet':
        if config['model']['arch'] == 'v2_x0_5':
            net = ShuffleNetV2.shufflenet_v2_x0_5(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x1_0':
            net = ShuffleNetV2.shufflenet_v2_x1_0(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x1_5':
            net = ShuffleNetV2.shufflenet_v2_x1_5(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x2_0':
            net = ShuffleNetV2.shufflenet_v2_x2_0(pretrained=False, progress=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'mobilenet':
        if config['model']['arch'] == 'small_075':
            net = Mobilenetv3.mobilenetv3_small_075(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'small_100':
            net = Mobilenetv3.mobilenetv3_small_100(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'large_075':
            net = Mobilenetv3.mobilenetv3_large_075(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'large_100':
            net = Mobilenetv3.mobilenetv3_large_100(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'rexnet':
        if config['model']['arch'] == 'rexnet1.0x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=1.0)
        elif config['model']['arch'] == 'rexnet1.5x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=1.5)
        elif config['model']['arch'] == 'rexnet2.0x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=2.0)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'arcface':
        if config['model']['arch'] == 'arcface_resnet18':
            net = Arcface.arcmargin_resnet18(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout,
                                             scale=64.0, margin=0.25)
        elif config['model']['arch'] == 'arcface_resnet50':
            net = Arcface.arcmargin_resnet50(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout,
                                             scale=64.0, margin=0.25)
        elif config['model']['arch'] == 'arcface_resnext50':
            net = Arcface.arcmargin_resnext50_32x4d(pretrained=False, progress=False, num_classes=num_classes, dropout=dropout,
                                             scale=64.0, margin=0.25)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    else:
        raise ValueError('Unsupported architecture: ' + str(config['model']['type']))

    return net

if __name__ == '__main__':
    import torch

    config = {'model':
                  {'type':'arcface',
                   'arch':'arcface_resnet18'}}

    net = load_model(config, 100, dropout=[False, False, False, False, False])

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    out = net(torch.randn(10, 3, 32, 32), torch.randn(10))
    print(out.size())