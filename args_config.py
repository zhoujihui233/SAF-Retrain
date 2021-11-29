import argparse

parser = argparse.ArgumentParser()
# network parameter
parser.add_argument("--network", type=str, default='lenet')
parser.add_argument("--mode", type=str, default='test')
parser.add_argument('--batch_size', type=int, default=64,
                    help='test batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')

# device parameter
parser.add_argument('--bit_width', type=int, default=8,
                    help='quantization, number of ReRAMs for a weight in one crossbar')
parser.add_argument('--SA_ratio', type=float, default=0.2,
                    help='ratio of SAF (SA0==SA1)')
