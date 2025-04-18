import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

# from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
# from ..transforms._presets import ImageClassification, InterpolationMode
# from ..utils import _log_api_usage_once
# from ._api import WeightsEnum, Weights
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import handle_legacy_interface, _ovewrite_named_param, _make_divisible


__all__ = [
    "RegNet",
    "RegNet_Y_400MF_Weights",
    "RegNet_Y_800MF_Weights",
    "RegNet_Y_1_6GF_Weights",
    "RegNet_Y_3_2GF_Weights",
    "RegNet_Y_8GF_Weights",
    "RegNet_Y_16GF_Weights",
    "RegNet_Y_32GF_Weights",
    "RegNet_Y_128GF_Weights",
    "RegNet_X_400MF_Weights",
    "RegNet_X_800MF_Weights",
    "RegNet_X_1_6GF_Weights",
    "RegNet_X_3_2GF_Weights",
    "RegNet_X_8GF_Weights",
    "RegNet_X_16GF_Weights",
    "RegNet_X_32GF_Weights",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_y_128gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
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




class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = Conv2dNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = Conv2dNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def _regnet(
    arch: str,
    block_params: BlockParams,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> RegNet:

    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model



def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (RegNet_Y_400MF_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet('regnet_y_400mf', params, pretrained, progress, **kwargs)



# def regnet_y_800mf(*, weights: Optional[RegNet_Y_800MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_800MF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_800MF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_800MF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_1_6gf(*, weights: Optional[RegNet_Y_1_6GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_1.6GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_1_6GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_1_6GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_3_2gf(*, weights: Optional[RegNet_Y_3_2GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_3.2GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_3_2GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_3_2GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_8gf(*, weights: Optional[RegNet_Y_8GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_8GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_8GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_8GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_16gf(*, weights: Optional[RegNet_Y_16GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_16GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_16GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_16GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_32gf(*, weights: Optional[RegNet_Y_32GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_32GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_Y_32GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_32GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_y_128gf(*, weights: Optional[RegNet_Y_128GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetY_128GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.
#     NOTE: Pretrained weights are not available for this model.

#     Args:
#         weights (RegNet_Y_128GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_Y_128GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(
#         depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs
#     )
#     return _regnet(params, weights, progress, **kwargs)



def regnet_x_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (RegNet_X_400MF_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    

    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet('regnet_x_400mf',params, pretrained, progress, **kwargs)



# def regnet_x_800mf(*, weights: Optional[RegNet_X_800MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_800MF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_800MF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_800MF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_x_1_6gf(*, weights: Optional[RegNet_X_1_6GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_1.6GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_1_6GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_1_6GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_x_3_2gf(*, weights: Optional[RegNet_X_3_2GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_3.2GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_3_2GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_3_2GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_x_8gf(*, weights: Optional[RegNet_X_8GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_8GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_8GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_8GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_x_16gf(*, weights: Optional[RegNet_X_16GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_16GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_16GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_16GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)



# def regnet_x_32gf(*, weights: Optional[RegNet_X_32GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
#     """
#     Constructs a RegNetX_32GF architecture from
#     `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

#     Args:
#         weights (RegNet_X_32GF_Weights, optional): The pretrained weights for the model
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     weights = RegNet_X_32GF_Weights.verify(weights)

#     params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
#     return _regnet(params, weights, progress, **kwargs)
