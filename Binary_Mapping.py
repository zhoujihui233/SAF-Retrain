from SAF import *
import torch
from args_config import parser

args = parser.parse_args()
bit_width = args.bit_width
SA0_ratio = args.SA_ratio
SA1_ratio = args.SA_ratio


def intArrToBinXB(int_arr):
    weight_xb = np.zeros(shape=(int_arr.shape[0], int_arr.shape[1], bit_width), dtype=int)
    for bit in range(bit_width):
        weight_xb[:, :, bit_width - 1 - bit] = (int_arr >> bit) & 1
    return weight_xb


def binXbToDecArr(bin_xb):
    coefficient = bit_width - 1 - np.array(range(bit_width))
    coefficient = np.power(2, coefficient)
    weight_arr = np.zeros(shape=(bin_xb.shape[0], bin_xb.shape[1]))
    for bit in range(bit_width):
        weight_arr += bin_xb[:, :, bit] * coefficient[bit]
    return weight_arr


def binary_mapping(name, weight):
    # print(name)
    if 'conv' in name:  # conv layer
        [m, n, p, q] = weight.shape
        weight_2d = weight.reshape(weight.shape[0], -1)
        weight_np = weight_2d.detach().cpu().numpy()
        weight_np = weight_np.astype(np.int32)
        if np.max(weight_np) >= np.power(2, bit_width):
            print(name, ':max value is out of range')
        if np.min(weight_np) < 0:
            print(name, ':min value is out of range')
        weight_xb = intArrToBinXB(weight_np)
        weight_xb = SA(name, weight_xb)
        if not os.path.exists('./data/'+args.network+'_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy'):
            alternative_set(name, weight_xb)
        weight_updated = binXbToDecArr(weight_xb)
        weight_updated = weight_updated.reshape(m, n, p, q)
        weight_updated = torch.from_numpy(weight_updated).cuda()
        return weight_updated
    elif 'fc' in name or 'linear' in name:  # fc layer
        weight_np = weight.detach().cpu().numpy()
        weight_np = weight_np.astype(np.int32)
        if np.max(abs(weight_np)) >= np.power(2, bit_width):
            print(name, ':max value is out of range ')
        if np.min(weight_np) < 0:
            print(name, ':min value is out of range')
        weight_xb = intArrToBinXB(weight_np)
        weight_xb = SA(name, weight_xb)
        if not os.path.exists('./data/'+args.network+'_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy'):
            alternative_set(name, weight_xb)
        weight_updated = binXbToDecArr(weight_xb)
        weight_updated = torch.from_numpy(weight_updated).cuda()
        return weight_updated
