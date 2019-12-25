import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import time
import nibabel
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.models as models
from torch.autograd import Variable
import random
from torchvision import transforms
from PIL import Image
from Epi.DatasetEPI import EPI, EPIPositive
from unet.unet_parts import ASPPBottleneck
from unet.unet_model import DepthUNet, UNet, CNN, NLFFUNet
# from eval import eval_net
# from dice_loss import dice_coeff
import warnings
import logging
import os

warnings.filterwarnings('ignore')
device = 0


class DSC_loss(nn.Module):
    def __init__(self):
        super(DSC_loss, self).__init__()

    def forward(self, img, target, smooth=1.):
        inter = torch.dot(img.view(-1), target.view(-1))
        union = torch.sum(img) + torch.sum(target) + smooth
        t = ((2 * inter.float()) + smooth) / union.float()

        return 1 - t


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def dice_coeff(input, target):
    """Dice coeff for batches"""

    s = DSC_loss().forward(input, target, smooth=1e-7)

    return 1 - s.item()


def tensor_max(input):
    input = input.view(-1)
    L = input.size(0)
    maximum = input[0]
    index = 0
    for i in range(L):
        if input[i] > maximum:
            maximum = input[i]
            index = i
    return maximum, index


def eval_net(net, dataset):
    net.eval()
    dice = 0
    for i in range(dataset.__len__()):
        img, true_mask = dataset[i]
        img = img.unsqueeze(0)
        img = img.cuda(device)
        true_mask = true_mask.cuda()
        mask_pred = net(img)
        # mask_pred = F.sigmoid(mask_pred)

        dice += dice_coeff(mask_pred, true_mask)
    return dice / (dataset.__len__())


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=True,):

    dir_img = "D:" + os.sep + "epi" + os.sep + "all" + os.sep
    dir_mask = "D:" + os.sep + "epi" + os.sep + "masks" + os.sep
    dir_val = "D:" + os.sep + "epi" + os.sep + "all" + os.sep
    dir_test = "D:" + os.sep + "epi" + os.sep + "test" + os.sep

    train = EPIPositive(image_paths=dir_img, target_paths=dir_mask,  train=True)
    val = EPIPositive(image_paths=dir_val, target_paths=dir_mask, train=False)
    test = EPIPositive(image_paths=dir_test, target_paths=dir_mask, train=False)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Test size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, train.__len__(),
               val.__len__(), test.__len__(), str(save_cp), str(gpu)))
    # optimizer
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=0.005)

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=0.005)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # loss function
    criterion_BCE = nn.BCELoss()
    criterion_DSC = DSC_loss()
    criterion_Focal = FocalLoss()
    criterion = nn.CrossEntropyLoss()

    test_dice = []
    val_dice = []
    BestDice = 0
    for epoch in tqdm(range(epochs)):
        train = EPIPositive(image_paths=dir_img, target_paths=dir_mask, train=True)
        val = EPIPositive(image_paths=dir_val, target_paths=dir_mask, train=False)
        train_loader = data.DataLoader(train, batch_size=1, shuffle=True, drop_last=True)
        test = EPIPositive(image_paths=dir_test, target_paths=dir_mask, train=False)
        net.train()
        epoch_loss = 0
        for i in range(train.__len__()):
            img, true_mask = train[i]
            img = img.unsqueeze(0)
            net.train()
        # for img, true_mask in train_loader:
            if gpu:
                img = img.cuda(device)
                true_mask = true_mask.cuda(device)
            mask_pred = net(img)
            mask_pred = mask_pred.contiguous()

            loss = 0
            loss += criterion_BCE(mask_pred.view(-1), true_mask.view(-1))
            # loss += criterion_DSC(mask_pred, true_mask)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\nEpoch{} finished ! Loss: {}'.format(epoch+1, epoch_loss / train.__len__()))
        # end = time.time()
        # print("time={}s".format(end-start))
        scheduler.step(epoch)
        # scheduler.step(epoch_loss / train.__len__())
        logging.info('Epoch {} Loss: {}'.format(epoch+1, epoch_loss / train.__len__()))
        val_dice_item = eval_net(net, val)
        val_dice.append(val_dice_item)
        print('Validation Dice Coeff: {}'.format(val_dice_item))
        logging.info('Validation: {}'.format(val_dice_item))
        # print('Test Dice Coeff: {}'.format(test_dice_item))
        # logging.info('Test: {}'.format(test_dice_item))
        logging.info('\n')
        print(optimizer.param_groups[0]['lr'])
        if val_dice_item >= BestDice:
            torch.save(net.state_dict(),
                       "." + 'CP{}.pth'.format(epoch + 1))
            BestDice = val_dice_item
        elif epoch % 10 == 1:
            torch.save(net.state_dict(),
                       "." + 'CP{}.pth'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default='.CP9.pth', help='load file model')
    parser.add_option('-n', '--net', dest='net', default='UNet')
    parser.add_option('-d', '--device', dest='device', default=0, type='int',
                      help='GPU device')
    parser.add_option('--log', dest='log', default='test_log.txt')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(filename=os.path.join(os.getcwd(), args.net+'.txt'),
                        level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    device = args.device
    net_choice = args.net
    # network = DepthUNet(n_classes=1, n_channels=3)
    network = NLFFUNet(n_channels=3, n_classes=1)
    # network = ASPPBottleneck(in_channels=3, out_channels=1, basic_channels=64)

    if args.load:
        network.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        network.cuda(device)
        cudnn.benchmark = False  # faster convolutions, but more memory

    try:
        train_net(net=network,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




