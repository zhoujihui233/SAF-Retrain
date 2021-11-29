import numpy as np
import os
from args_config import parser
import sys

args = parser.parse_args()
bit_width = args.bit_width
SA0_ratio = args.SA_ratio
SA1_ratio = args.SA_ratio


def SA_index(name, weight_xb):
    index = np.random.randint(0, high=weight_xb.size, size=int((SA1_ratio + SA0_ratio) * weight_xb.size))
    SA1_index = index[:int(SA1_ratio * weight_xb.size)]
    SA0_index = index[int(SA1_ratio * weight_xb.size):]
    np.save('./data/'+args.network+'_SA1_index_' + str(SA1_ratio) + '_' + name + '.npy', SA1_index)
    np.save('./data/'+args.network+'_SA0_index_' + str(SA0_ratio) + '_' + name + '.npy', SA0_index)


def SA(name, weight_xb):
    row = weight_xb.shape[0]
    col = weight_xb.shape[1]
    weight_xb = weight_xb.reshape(row * col * bit_width)
    if not os.path.exists('./data/'+args.network+'_SA1_index_' + str(SA1_ratio) + '_' + name + '.npy') or \
            not os.path.exists('./data/'+args.network+'_SA0_index_' + str(SA0_ratio) + '_' + name + '.npy'):
        SA_index(name, weight_xb)
    SA1_index = np.load('./data/'+args.network+'_SA1_index_' + str(SA1_ratio) + '_' + name + '.npy')
    SA0_index = np.load('./data/'+args.network+'_SA0_index_' + str(SA0_ratio) + '_' + name + '.npy')
    weight_xb[SA1_index] = 1
    weight_xb[SA0_index] = 0
    weight_xb = weight_xb.reshape(row, col, bit_width)
    return weight_xb


def intArrToBin(int_arr):
    weight_xb = np.zeros(shape=(int_arr.size, bit_width), dtype=int)
    for bit in range(bit_width):
        weight_xb[:, bit_width - 1 - bit] = (int_arr >> bit) & 1
    return weight_xb


def alternative_set(name, weight_xb):
    row = weight_xb.shape[0]
    col = weight_xb.shape[1]
    SA1_index = np.load('./data/' + args.network + '_SA1_index_' + str(SA1_ratio) + '_' + name + '.npy')
    SA0_index = np.load('./data/' + args.network + '_SA0_index_' + str(SA0_ratio) + '_' + name + '.npy')
    weight_saf = -np.ones(shape=weight_xb.size)
    weight_saf[SA1_index] = 1
    weight_saf[SA0_index] = 0
    weight_saf = weight_saf.reshape(row, col, bit_width)
    index0 = np.argwhere(weight_saf == 0)
    index1 = np.argwhere(weight_saf == 1)
    if not os.path.exists('./data/binary_weight_' + str(bit_width) + 'bit_LUT.npy'):
        # binary_weight_LUT = np.zeros(shape=(2 ** bit_width, bit_width))
        w = np.linspace(0, 2 ** bit_width - 1, num=2 ** bit_width, dtype=int)
        binary_weight_LUT = intArrToBin(w)
        np.save('./data/binary_weight_' + str(bit_width) + 'bit_LUT.npy', binary_weight_LUT)
    else:
        binary_weight_LUT = np.load('./data/binary_weight_' + str(bit_width) + 'bit_LUT.npy')
    dic1 = {}
    key = 0
    for i in index0:
        if key == [i[0], i[1]]:
            for w in range(2 ** bit_width):
                bit_str = binary_weight_LUT[w]
                if bit_str[i[2]] == 1 and w in temp_list:
                    temp_list.remove(w)
        else:
            key = [i[0], i[1]]
            temp_list = []
            for w in range(2 ** bit_width):
                bit_str = binary_weight_LUT[w]
                if bit_str[i[2]] == 0:
                    temp_list.append(w)
        dic1[tuple(key)] = temp_list
    dic2 = {}
    key = 0
    for i in index1:
        if key == [i[0], i[1]]:
            for w in range(2 ** bit_width):
                bit_str = binary_weight_LUT[w]
                if bit_str[i[2]] == 0 and w in temp_list:
                    temp_list.remove(w)
        else:
            key = [i[0], i[1]]
            temp_list = []
            for w in range(2 ** bit_width):
                bit_str = binary_weight_LUT[w]
                if bit_str[i[2]] == 1:
                    temp_list.append(w)
        dic2[tuple(key)] = temp_list
    dic = {}
    keys = dic1.keys() | dic2.keys()
    for key in keys:
        if key in dic1 and key in dic2:
            dic[key] = list(set(dic1[key]) & (set(dic2[key])))
        elif key in dic1:
            dic[key] = dic1[key]
        else:
            dic[key] = dic2[key]
    np.save('./data/'+args.network+'_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy', dic)


if __name__ == '__main__':
    arr = np.random.randint(1, 20, size=(2, 5, 4))
    # SA_index(arr, 0.1, 0.2)

    # weight_saf = -np.ones(shape=arr.shape)
    # weight_saf[0, 2, 3] = 1
    # weight_saf[0, 4, 1] = 1
    # index = np.argwhere(weight_saf == 1)
    # # weight_saf[index] = 2
    # # print(index)
    # for i in index:
    #     print(i)
    #     print(i[2])
