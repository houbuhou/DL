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
from torch.autograd import Variable
import random
from torchvision import transforms
from PIL import Image
from Epi.DatasetEPI import EPI
from unet.unet_parts import ASPPBottleneck

# from eval import eval_net
from unet import UNet, R_ResUNet
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
        [C, H, W] = target.shape
        self.inter = torch.dot(img.view(-1), target.view(-1))
        self.union = torch.sum(img) + torch.sum(target) + smooth

        self.img_0 = 1 - img
        self.target_0 = 1 - target

        self.inter_0 = torch.dot(self.img_0.view(-1), self.target_0.view(-1))
        self.union_0 = torch.sum(self.img_0) + torch.sum(self.target_0) + smooth

        t = ((2 * self.inter.float()) + smooth) / self.union.float()
        t_0 = ((2 * self.inter_0.float()) + smooth) / self.union_0.float()
        obj_dice = t

        return 1 - obj_dice.item()


def dice_coeff(input, target):
    """Dice coeff for batches"""

    s = DSC_loss().forward(input, target, smooth=1e-7)

    return 1 - s


def eval_net(net, dataset):
    net.eval()
    dice = 0
    for i in range(dataset.__len__()):
        img, true_mask = dataset[i]
        img = img.unsqueeze(0)
        img = img.cuda(device)
        true_mask = true_mask.cuda(device)
        mask_pred = net(img)
        mask_pred = mask_pred.contiguous()
        # mask_pred = F.sigmoid(mask_pred)

        mask_pred_show = mask_pred.detach().cpu().numpy()
        mask_pred_show = mask_pred_show.squeeze().astype(np.float32)
        # plt.figure(0)
        # plt.imshow(mask_pred_show)
        # plt.figure(1)
        # plt.imshow(true_mask.cpu().squeeze())
        # plt.figure(2)
        # img_show = img.squeeze().cpu()
        # plt.imshow(img_show[0])
        # # plt.figure(3)
        # # plt.hist(mask_pred_show)
        # plt.show()


        # To normalize predict mask
        # mask_pred = F.sigmoid(mask_pred)

        dice += dice_coeff(mask_pred, true_mask)
    return dice / (dataset.__len__())


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=True,):

    dir_img = "D:" + os.sep + "epi" + os.sep + "train" + os.sep
    dir_mask = "D:" + os.sep + "epi" + os.sep + "masks" + os.sep
    dir_val = "D:" + os.sep + "epi" + os.sep + "val" + os.sep
    dir_test = "D:" + os.sep + "epi" + os.sep + "test" + os.sep

    train = EPI(image_paths=dir_img, target_paths=dir_mask,  train=True)
    val = EPI(image_paths=dir_val, target_paths=dir_mask, train=False)
    test = EPI(image_paths=dir_test, target_paths=dir_mask, train=False)

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
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.0005)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # loss function
    criterion_BCE = nn.BCELoss()
    criterion_DSC = DSC_loss()

    test_dice = []
    val_dice = []
    BestDice = 0
    for epoch in tqdm(range(epochs)):
        train = EPI(image_paths=dir_img, target_paths=dir_mask, train=True)
        val = EPI(image_paths=dir_val, target_paths=dir_mask, train=False)
        train_loader = data.DataLoader(train, batch_size=2, shuffle=True, drop_last=True)
        test = EPI(image_paths=dir_test, target_paths=dir_mask, train=False)
        # print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # start = time.time()
        epoch_loss = 0
        for i in range(train.__len__()):
            img, true_mask = train[i]
            img = img.unsqueeze(0)
        # for img, true_mask in train_loader:
            if gpu:
                img = img.cuda(device)
                true_mask = true_mask.cuda(device)
            mask_pred = net(img)
            net.train()
            mask_pred = mask_pred.contiguous()
            mask_pred = F.sigmoid(mask_pred)   # for DeepLab and FCN
            # print(mask_pred.shape)
            mask_prob_flat = mask_pred.view(-1)
            true_mask_flat = true_mask.view(-1)
            loss = 0
            # loss = criterion_BCE(mask_prob_flat, true_mask_flat)
            loss += criterion_BCE(mask_prob_flat, true_mask_flat)
            # loss += criterion_DSC(mask_pred[1], true_mask[1])
            # print(loss)
            epoch_loss += loss.item()

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / train.__len__(), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if(loss.item() > -1):
            #     mask_pred_show = mask_pred.detach().cpu().numpy()
            #     mask_pred_show = mask_pred_show.squeeze().astype(np.float32)
            #     plt.figure(0)
            #     plt.imshow(mask_pred_show)
            #     plt.figure(1)
            #     plt.imshow(true_mask.cpu().squeeze())
            #     plt.figure(2)
            #     img_show = img.squeeze().cpu()
            #     plt.imshow(img_show[0])
            #     plt.show()
        print('\nEpoch{} finished ! Loss: {}'.format(epoch+1, epoch_loss / train.__len__()))
        # end = time.time()
        # print("time={}s".format(end-start))
        scheduler.step(epoch)
        # scheduler.step(epoch_loss / train.__len__())
        logging.info('Epoch {} Loss: {}'.format(epoch+1, epoch_loss / train.__len__()))
        test_dice_item = eval_net(net, test)
        test_dice.append(test_dice_item)
        val_dice_item = eval_net(net, val)
        val_dice.append(val_dice_item)
        print('Validation Dice Coeff: {}'.format(val_dice_item))
        logging.info('Validation: {}'.format(val_dice_item))
        print('Test Dice Coeff: {}'.format(test_dice_item))
        logging.info('Test: {}'.format(test_dice_item))
        logging.info('\n')
        if val_dice_item >= BestDice:
            torch.save(net.state_dict(),
                       "." + 'CP{}.pth'.format(epoch + 1))
            BestDice = val_dice_item
        # print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
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
    # network = UNet(n_classes=1, n_channels=3)
    network = ASPPBottleneck(in_channels=3, out_channels=1, basic_channels=64)

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




