import torch.nn as nn
from SAF import *
from quantization import *
from Binary_Mapping import *
from model_reprocess import *

criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted


def test_acc():
    eval_loss = 0
    eval_acc = 0
    for (test_img, test_label) in test_loader:
        test_img = test_img.cuda()
        test_label = test_label.cuda()

        test_out = model(test_img)
        loss = criterion(test_out, test_label)
        eval_loss += loss.data.item()
        _, pred = torch.max(test_out, 1)
        num_correct = (pred == test_label).sum()
        eval_acc += num_correct.item()  # Torch -> dictionary
    eval_acc = eval_acc / len(test_loader.dataset)
    print("TEST total loss: ", eval_loss)
    print("TEST acc: ", eval_acc)


def compute_net():
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if 'conv' in name or 'linear' in name or 'fc' in name:
                # print(name)
                scale, zero_point = weight_quantize_parameter(module.weight)
                weight_update = quantize(module.weight, scale, zero_point)
                weight_update = binary_mapping(name, weight_update)
                module.weight = torch.nn.Parameter(de_quantize(weight_update, scale, zero_point))


if __name__ == '__main__':
    print('network:', args.network, 'ratio of SAF:', args.SA_ratio*2)
    print('before retraining')
    model.load_state_dict(torch.load(filename))
    model.eval()
    compute_net()
    test_acc()
    print('after retraining')
    filename_retrain = './model/'+args.network+'_retrain_'+str(args.SA_ratio)+'.pth'
    model.load_state_dict(torch.load(filename_retrain))
    model.eval()
    compute_net()
    test_acc()
