import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
# from models import LeNet
import resnet
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        data = torch.autograd.Variable(data)
        target = torch.autograd.Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            target = target.cuda(async=True)
            # data = torch.autograd.Variable(data, volatile=True)
            # target = torch.autograd.Variable(target, volatile=True)

            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def cam(model, epoch):
    model.eval()
    images_prefix = 'imgs/{:d}.jpg'
    global feature_blob

    with torch.no_grad():
        for img in xrange(10):

            para = list(model.parameters())[-2][img:img+1]
            para = para.data.cpu().numpy()

            image = Image.open(images_prefix.format(img))
            transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                        ])

            tensor = transform(image)
            tensor = tensor.view(1, 1, 28, 28)
            model(tensor)

            cam_feat = feature_blob[0].view(16, -1).data.cpu().numpy()
            cam = np.matmul(para, cam_feat)[0].reshape(8, 8)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam = cv2.resize(cam_img, (28, 28))
            heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
            image = cv2.imread(images_prefix.format(img))
            save_img = heatmap*0.3 + image*0.5
            save_img = cv2.resize(save_img, (224, 224))
            cv2.imwrite('result/cam_{}_{}.jpg'.format(img, epoch), save_img)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    feature_blob = np.zeros([1, 512, 4, 4])

    model = resnet.ResNet18()
    def hook(module, input, output):
        global feature_blob
        feature_blob = output

    model._modules.get('conv2').register_forward_hook(hook)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        cam(model, epoch)
        test(args, model, test_loader)


if __name__ == '__main__':
    main()
