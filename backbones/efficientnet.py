import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, List, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
# from ..transforms._presets import ImageClassification, InterpolationMode
# from ..utils import _log_api_usage_once
# from ._api import WeightsEnum, Weights
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import handle_legacy_interface, _ovewrite_named_param, _make_divisible


__all__ = [
    "EfficientNet",
    "EfficientNet_B0_Weights",
    "EfficientNet_B1_Weights",
    "EfficientNet_B2_Weights",
    "EfficientNet_B3_Weights",
    "EfficientNet_B4_Weights",
    "EfficientNet_B5_Weights",
    "EfficientNet_B6_Weights",
    "EfficientNet_B7_Weights",
    "EfficientNet_V2_S_Weights",
    "EfficientNet_V2_M_Weights",
    "EfficientNet_V2_L_Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]



######### Add by me begin ######## 

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        # _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )

class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input        
        
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
      
        
######### Add by me end ######## 
        

    
@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        # _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    arch: str,
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int], # weights: Optional[WeightsEnum],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


# _COMMON_META = {
#     "task": "image_classification",
#     "categories": _IMAGENET_CATEGORIES,
#     "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet",
# }


# _COMMON_META_V1 = {
#     **_COMMON_META,
#     "architecture": "EfficientNet",
#     "publication_year": 2019,
#     "interpolation": InterpolationMode.BICUBIC,
#     "min_size": (1, 1),
# }


# _COMMON_META_V2 = {
#     **_COMMON_META,
#     "architecture": "EfficientNetV2",
#     "publication_year": 2021,
#     "interpolation": InterpolationMode.BILINEAR,
#     "min_size": (33, 33),
# }


# class EfficientNet_B0_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
#         transforms=partial(
#             ImageClassification, crop_size=224, resize_size=256, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 5288548,
#             "size": (224, 224),
#             "acc@1": 77.692,
#             "acc@5": 93.532,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B1_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
#         transforms=partial(
#             ImageClassification, crop_size=240, resize_size=256, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 7794184,
#             "size": (240, 240),
#             "acc@1": 78.642,
#             "acc@5": 94.186,
#         },
#     )
#     IMAGENET1K_V2 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
#         transforms=partial(
#             ImageClassification, crop_size=240, resize_size=255, interpolation=InterpolationMode.BILINEAR
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 7794184,
#             "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-lr-wd-crop-tuning",
#             "interpolation": InterpolationMode.BILINEAR,
#             "size": (240, 240),
#             "acc@1": 79.838,
#             "acc@5": 94.934,
#         },
#     )
#     DEFAULT = IMAGENET1K_V2


# class EfficientNet_B2_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
#         transforms=partial(
#             ImageClassification, crop_size=288, resize_size=288, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 9109994,
#             "size": (288, 288),
#             "acc@1": 80.608,
#             "acc@5": 95.310,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B3_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
#         transforms=partial(
#             ImageClassification, crop_size=300, resize_size=320, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 12233232,
#             "size": (300, 300),
#             "acc@1": 82.008,
#             "acc@5": 96.054,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B4_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
#         transforms=partial(
#             ImageClassification, crop_size=380, resize_size=384, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 19341616,
#             "size": (380, 380),
#             "acc@1": 83.384,
#             "acc@5": 96.594,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B5_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
#         transforms=partial(
#             ImageClassification, crop_size=456, resize_size=456, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 30389784,
#             "size": (456, 456),
#             "acc@1": 83.444,
#             "acc@5": 96.628,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B6_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
#         transforms=partial(
#             ImageClassification, crop_size=528, resize_size=528, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 43040704,
#             "size": (528, 528),
#             "acc@1": 84.008,
#             "acc@5": 96.916,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_B7_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
#         transforms=partial(
#             ImageClassification, crop_size=600, resize_size=600, interpolation=InterpolationMode.BICUBIC
#         ),
#         meta={
#             **_COMMON_META_V1,
#             "num_params": 66347960,
#             "size": (600, 600),
#             "acc@1": 84.122,
#             "acc@5": 96.908,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_V2_S_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
#         transforms=partial(
#             ImageClassification,
#             crop_size=384,
#             resize_size=384,
#             interpolation=InterpolationMode.BILINEAR,
#         ),
#         meta={
#             **_COMMON_META_V2,
#             "num_params": 21458488,
#             "size": (384, 384),
#             "acc@1": 84.228,
#             "acc@5": 96.878,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_V2_M_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
#         transforms=partial(
#             ImageClassification,
#             crop_size=480,
#             resize_size=480,
#             interpolation=InterpolationMode.BILINEAR,
#         ),
#         meta={
#             **_COMMON_META_V2,
#             "num_params": 54139356,
#             "size": (480, 480),
#             "acc@1": 85.112,
#             "acc@5": 97.156,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# class EfficientNet_V2_L_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
#         transforms=partial(
#             ImageClassification,
#             crop_size=480,
#             resize_size=480,
#             interpolation=InterpolationMode.BICUBIC,
#             mean=(0.5, 0.5, 0.5),
#             std=(0.5, 0.5, 0.5),
#         ),
#         meta={
#             **_COMMON_META_V2,
#             "num_params": 118515272,
#             "size": (480, 480),
#             "acc@1": 85.808,
#             "acc@5": 97.788,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1


# @handle_legacy_interface(weights=("pretrained", EfficientNet_B0_Weights.IMAGENET1K_V1))
def efficientnet_b0(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B0_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # weights = EfficientNet_B0_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet('efficientnet_b0',inverted_residual_setting, 0.2, last_channel, pretrained, progress, **kwargs)


# @handle_legacy_interface(weights=("pretrained", EfficientNet_B1_Weights.IMAGENET1K_V1))
# def efficientnet_b1(
#     *, weights: Optional[EfficientNet_B1_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B1 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B1_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B1_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
#     return _efficientnet(inverted_residual_setting, 0.2, last_channel, weights, progress, **kwargs)


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B2_Weights.IMAGENET1K_V1))
# def efficientnet_b2(
#     *, weights: Optional[EfficientNet_B2_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B2 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B2_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B2_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
#     return _efficientnet(inverted_residual_setting, 0.3, last_channel, weights, progress, **kwargs)


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B3_Weights.IMAGENET1K_V1))
# def efficientnet_b3(
#     *, weights: Optional[EfficientNet_B3_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B3 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B3_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B3_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
#     return _efficientnet(inverted_residual_setting, 0.3, last_channel, weights, progress, **kwargs)


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B4_Weights.IMAGENET1K_V1))
# def efficientnet_b4(
#     *, weights: Optional[EfficientNet_B4_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B4 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B4_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B4_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
#     return _efficientnet(inverted_residual_setting, 0.4, last_channel, weights, progress, **kwargs)


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B5_Weights.IMAGENET1K_V1))
# def efficientnet_b5(
#     *, weights: Optional[EfficientNet_B5_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B5 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B5_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B5_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
#     return _efficientnet(
#         inverted_residual_setting,
#         0.4,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
#         **kwargs,
#     )


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B6_Weights.IMAGENET1K_V1))
# def efficientnet_b6(
#     *, weights: Optional[EfficientNet_B6_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B6 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B6_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B6_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
#     return _efficientnet(
#         inverted_residual_setting,
#         0.5,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
#         **kwargs,
#     )


# # @handle_legacy_interface(weights=("pretrained", EfficientNet_B7_Weights.IMAGENET1K_V1))
# def efficientnet_b7(
#     *, weights: Optional[EfficientNet_B7_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs a EfficientNet B7 architecture from
#     `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

#     Args:
#         weights (EfficientNet_B7_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_B7_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
#     return _efficientnet(
#         inverted_residual_setting,
#         0.5,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
#         **kwargs,
#     )


# @handle_legacy_interface(weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1))
# def efficientnet_v2_s(
#     *, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs an EfficientNetV2-S architecture from
#     `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

#     Args:
#         weights (EfficientNet_V2_S_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_V2_S_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
#     return _efficientnet(
#         inverted_residual_setting,
#         0.2,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
#         **kwargs,
#     )


# @handle_legacy_interface(weights=("pretrained", EfficientNet_V2_M_Weights.IMAGENET1K_V1))
# def efficientnet_v2_m(
#     *, weights: Optional[EfficientNet_V2_M_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs an EfficientNetV2-M architecture from
#     `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

#     Args:
#         weights (EfficientNet_V2_M_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_V2_M_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
#     return _efficientnet(
#         inverted_residual_setting,
#         0.3,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
#         **kwargs,
#     )


# @handle_legacy_interface(weights=("pretrained", EfficientNet_V2_L_Weights.IMAGENET1K_V1))
# def efficientnet_v2_l(
#     *, weights: Optional[EfficientNet_V2_L_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> EfficientNet:
#     """
#     Constructs an EfficientNetV2-L architecture from
#     `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

#     Args:
#         weights (EfficientNet_V2_L_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = EfficientNet_V2_L_Weights.verify(weights)

#     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
#     return _efficientnet(
#         inverted_residual_setting,
#         0.4,
#         last_channel,
#         weights,
#         progress,
#         norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
#         **kwargs,
#     )
