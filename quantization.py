import torch
from args_config import parser

args = parser.parse_args()
bit_width = args.bit_width


# Quantization #############################################
def weight_quantize_parameter(weight):  # 0~2^bits-1
    max_w = weight.max().item()
    min_w = weight.min().item()
    level = 2 ** bit_width - 1
    scale = (max_w - min_w) / level
    zero_point = round((0.0 - min_w) / scale)
    return scale, zero_point


def quantize(weight, scale, zero_point):  # 0~2^bits-1
    return torch.clamp((weight / scale).round() + zero_point, 0, 2 ** bit_width - 1)
    return weight_quantized


def de_quantize(weight, scale, zero_point):
    # print('scale:', scale, 'zero point:', zero_point)
    weight_dequantized = scale * (weight - zero_point).float()
    return weight_dequantized
