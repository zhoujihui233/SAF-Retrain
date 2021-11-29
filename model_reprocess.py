from lenet import LeNet
from resnet import ResNet18
from alexnet import AlexNet
import torch
import torchvision
import torch.utils.data as Data
from torchvision import datasets, transforms
from args_config import parser

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


if args.network == 'lenet':
    model = LeNet()
    train_dataset = torchvision.datasets.MNIST(root='E:/PycharmProjects/NNtraining', train=True,
                                               transform=torchvision.transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.MNIST(root='E:/PycharmProjects/NNtraining', train=False,
                                              transform=torchvision.transforms.ToTensor(), )

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    filename = 'lenet_0.9917.pth'
elif args.network == 'alexnet':
    model = AlexNet()
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('E:/PycharmProjects/NNtraining/CIFAR10', train=True, download=False,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('E:/PycharmProjects/NNtraining/CIFAR10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    filename = 'alexnet_cifar_87.39.pth'
model.cuda()
