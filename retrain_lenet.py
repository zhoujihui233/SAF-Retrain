from SAF import *
import torch.optim as optim
from model_reprocess import *
import torch.nn as nn
from args_config import parser
import os
import random
from quantization import *
from Binary_Mapping import *

args = parser.parse_args()
SA0_ratio = args.SA_ratio
SA1_ratio = args.SA_ratio

criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted
test_metrics = []


def random_fix(weight_2d, dic):
    for key in dic:
        alter_set = np.array(dic[key])
        if weight_2d[key] in alter_set:
            continue
        else:
            distance = np.array(dic[key])
            distance = abs(distance - weight_2d[key])
            prob = 1. / distance
            prob = prob / prob.sum()
            a = random.random()
            b = 0
            for i in range(prob.size):
                if b <= a < b + prob[i]:
                    weight_2d[key] = alter_set[i]
                else:
                    b = b + prob[i]
    return weight_2d


def mask_retrain(acc_original):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = args.epochs
    log_batch = 30
    # train_metrics = []
    dic = {}
    dic_remnant = {}
    for name, parms in model.named_parameters():
        if ('conv' in name or 'linear' in name or 'fc' in name) and 'weight' in name:
            name = name.strip('.weight')
            print(name)
            if not os.path.exists('./data/' + args.network + '_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy'):
                print('alternative set does not exist in' + name)
            dic[name] = np.load('./data/' + args.network + '_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy',
                                allow_pickle=True).item()
            dic_remnant[name] = np.load('./data/' + args.network + '_AlterSet_' + str(SA1_ratio) + '_' + name + '.npy',
                                        allow_pickle=True).item()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_classified = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            if i == 1 and epoch % 2 == 0:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        if 'conv' in name or 'linear' in name or 'fc' in name:
                            if dic_remnant[name]:
                                cnt = 0
                                dic_temp = {}
                                for key in dic_remnant[name]:
                                    cnt += 1
                                    dic_temp[key] = dic_remnant[name][key]
                                    if cnt > 0.1 * len(dic[name]):
                                        break
                                for key in dic_temp:
                                    dic_remnant[name].pop(key)
                                scale, zero_point = weight_quantize_parameter(module.weight)
                                weight = quantize(module.weight, scale, zero_point)
                                weight_2d = weight.reshape(module.weight.shape[0], -1).detach().cpu().numpy()
                                weight_2d = random_fix(weight_2d, dic_temp)
                                weight_updated = weight_2d.reshape(module.weight.shape)
                                weight_updated = torch.from_numpy(weight_updated).cuda()
                                module.weight = torch.nn.Parameter(de_quantize(weight_updated, scale, zero_point))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, parms in model.named_parameters():
                if ('conv' in name or 'linear' in name or 'fc' in name) and 'weight' in name:
                    name = name.strip('.weight')
                    grad = parms.grad
                    grad_2d = grad.detach().cpu().reshape(grad.shape[0], -1).numpy()
                    for key in dic[name]:
                        if key not in dic_remnant[name]:
                            grad_2d[key] = 0
                    grad = grad_2d.reshape(grad.shape)
                    parms.grad = torch.from_numpy(grad).cuda()
            # if epoch == (epochs-1) and i % log_batch == log_batch - 1:
            #     for name, module in model.named_modules():
            #         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            #             if 'conv' in name or 'linear' in name or 'fc' in name:
            #                 scale, zero_point = weight_quantize_parameter(module.weight)
            #                 weight = quantize(module.weight, scale, zero_point)
            #                 weight_2d = weight.reshape(module.weight.shape[0], -1).detach().cpu().numpy()
            #                 for key in dic[name]:
            #                     if weight_2d[key] not in dic[name][key]:
            #                         print('error!:', name, key)
            #                         print(weight_2d[key])
            #                         print(dic[name][key])
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct_classified += (predicted == labels).sum().item()
            running_loss += loss.item()
            if i % log_batch == log_batch - 1:
                avg_loss = running_loss / log_batch
                print('Epoch: %d/%d Batch: %5d loss: %.3f' % (epoch + 1, epochs, i + 1, avg_loss))
                running_loss = 0.0
        train_acc = (correct_classified / len(train_loader.dataset))
        # train_metrics.append(train_acc)
        print('Train accuracy of the network images: %.2f %%' % float(100 * train_acc))
        correct_classified = 0
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    if 'conv' in name or 'linear' in name or 'fc' in name:
                        scale, zero_point = weight_quantize_parameter(module.weight)
                        weight_update = quantize(module.weight, scale, zero_point)
                        weight_update = binary_mapping(name, weight_update)
                        module.weight = torch.nn.Parameter(de_quantize(weight_update, scale, zero_point))
            for data in test_loader:
                images, labels = data
                inputs, labels = images.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct_classified += (predicted == labels).sum().item()
            test_acc = (correct_classified / len(test_loader.dataset))
            print('Test accuracy of the network: %.2f %%' % float(100 * test_acc))
            if test_acc > acc_original and test_acc > 0.9:
                torch.save(model.state_dict(), "./model/" + args.network + "_retrain_" + str(SA1_ratio) + '_'
                           + str(test_acc) + '.pth')
                acc_original = test_acc
    print('acc after retrain:', acc_original)
    test_metrics.append(acc_original)


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
                scale, zero_point = weight_quantize_parameter(module.weight)
                weight_update = quantize(module.weight, scale, zero_point)
                weight_update = binary_mapping(name, weight_update)
                module.weight = torch.nn.Parameter(de_quantize(weight_update, scale, zero_point))


if __name__ == '__main__':
    print(args.network)
    model.load_state_dict(torch.load(filename))
    model.eval()
    mask_retrain(0.5)
    # "./model/" + args.network + "_retrain_" + str(SA1_ratio) + '_.pth'
    compute_net()
    # test_acc()
