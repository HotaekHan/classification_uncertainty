
""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet_blocks import round_channels, resolve_bn_args, resolve_act_layer
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights
from .layers.create_conv2d import create_conv2d
from .layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from .layers.activations import hard_sigmoid

__all__ = ['MobileNetV3']

# def _cfg(url='', **kwargs):
#     return {
#         'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
#         'crop_pct': 0.875, 'interpolation': 'bilinear',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'conv_stem', 'classifier': 'classifier',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'mobilenetv3_large_075': _cfg(url=''),
#     'mobilenetv3_large_100': _cfg(
#         interpolation='bicubic',
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
#     'mobilenetv3_small_075': _cfg(url='')
#     'mobilenetv3_small_100': _cfg(url='')
# }

_DEBUG = False

class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(MobileNetV3, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_model(model_kwargs, pretrained=False):
    if model_kwargs.pop('features_only', False):
        # load_strict = False
        # model_kwargs.pop('num_classes', 0)
        # model_kwargs.pop('num_features', 0)
        # model_kwargs.pop('head_conv', None)
        # model_class = MobileNetV3Features
        raise NotImplementedError
    else:
        load_strict = True
        model_class = MobileNetV3

    model = model_class(**model_kwargs)
    # model.default_cfg = default_cfg
    if pretrained:
        # load_pretrained(
        #     model,
        #     default_cfg,
        #     num_classes=model_kwargs.get('num_classes', 0),
        #     in_chans=model_kwargs.get('in_chans', 3),
        #     strict=load_strict)
        raise NotImplementedError
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024

        act_layer = resolve_act_layer(kwargs, 'hard_swish')
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
            # stage 1, 56x56 in
            ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
            # stage 2, 28x28 in
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
            # stage 3, 14x14 in
            ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
            # stage 4, 14x14in
            ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c576'],  # hard-swish
        ]
    else:
        num_features = 1280

        act_layer = resolve_act_layer(kwargs, 'hard_swish')
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_nre'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
            # stage 2, 56x56 in
            ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
            # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
            # stage 4, 14x14in
            ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
            # stage 5, 14x14in
            ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c960'],  # hard-swish
        ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    model = _create_model(model_kwargs, pretrained)
    return model


def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model

def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model

def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model

def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model
