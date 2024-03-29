import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import time
import utils
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

# from eval import eval_net
from unet import UNet, R_ResUNet
from network import U_Net, R2AttU_Net, R2U_Net
from model import FCN8s

from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
# from dice_loss import dice_coeff
import warnings
import logging
import os

from DatasetEPI import EPI
from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback


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
        obj_dice = (t + t_0)/2
        # if(torch.sum(target) == 0):
        #     t = 1 - (torch.sum(img) / (512 * 512))

        return 1 - obj_dice


def dice_coeff(input, target):
    """Dice coeff for batches"""

    s = DSC_loss().forward(input, target, smooth=1e-7)
    # if(s<0.2):
    #     print(s)
    #     plt.figure(0)
    #     plt.imshow(input.cpu().squeeze())
    #     plt.figure(1)
    #     plt.imshow(target.cpu().squeeze())
    #     plt.show()

    return 1 - s


class NiiDataset(data.Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.files = os.listdir(self.image_paths)
        self.labels = os.listdir(self.target_paths)

    def transform(self, image, mask):

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        # Resize
        resize = transforms.Resize(size=(64, 64))
        image = resize(image)
        mask = resize(mask)

        # Color jitter
        trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.25, saturation=0.25, contrast=0.25)
        ])
        if random.random() > 0.5:
            image = trans(image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(64, 64))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        image_data = nibabel.load(self.image_paths + self.files[index])
        image = image_data.get_fdata()
        mask_data = nibabel.load(self.target_paths + self.files[index])
        mask = mask_data.get_fdata()
        shape = image.shape
        self.x = []
        self.y = []
        for i in range(shape[2]):
            image_2D = image[i]
            label_2D = mask[i]
            image_3D = np.expand_dims(image_2D, axis=0)
            label_3D = np.expand_dims(label_2D, axis=0)
            image_T, label_T = self.transform(image_3D, label_3D)
            self.x.append(image_T)
            self.y.append(label_T)
        return self.x, self.y

    def __len__(self):
        return len(self.files)

def eval_net(net, dataset):
    net.eval()
    dice = 0
    num = 0
    for i in range(dataset.__len__()):
        image_list, true_mask_list = dataset[i]
        for j in range(len(image_list)):
            img = image_list[j]
            true_mask = true_mask_list[j]
            img = img.unsqueeze(0)
            img = img.cuda(device)
            true_mask = true_mask.cuda(device)
            mask_pred = net(img)
            mask_pred = mask_pred.contiguous()

            mask_pred_show = mask_pred.detach().cpu().numpy()
            mask_pred_show = mask_pred_show.squeeze().astype(np.float32)


            dice += dice_coeff(mask_pred, true_mask).item()
            num += 1
    return dice / (dataset.__len__())




def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=True,):

    TrainDir = "/home/cyf/Dataset/Task04_Hippocampus/imagesTr/"
    LableDir = "/home/cyf/Dataset/Task04_Hippocampus/labelsTr/"
    ValDir = "/home/cyf/Dataset/Task04_Hippocampus/imagesTr/"
    TestDir = "/home/cyf/Dataset/Task04_Hippocampus/imagesTr/"

    train = NiiDataset(image_paths=TrainDir, target_paths=LableDir,  train=True)
    val = NiiDataset(image_paths=ValDir, target_paths=LableDir, train=False)
    test = NiiDataset(image_paths=TestDir, target_paths=LableDir, train=False)

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # loss function
    criterion_BCE = nn.BCELoss()
    criterion_DSC = DSC_loss()

    test_dice = []
    val_dice = []
    best_dice = 0
    for epoch in tqdm(range(epochs)):
        train = NiiDataset(image_paths=TrainDir, target_paths=LableDir, train=True)
        val = NiiDataset(image_paths=ValDir, target_paths=LableDir, train=False)
        test = NiiDataset(image_paths=TestDir, target_paths=LableDir, train=False)
        net.train()
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
            mask_pred = F.sigmoid(mask_pred)
            loss = 0
            loss += criterion_DSC(mask_pred[1], true_mask[1])
            # print(loss.item())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\nEpoch{} finished ! Loss: {}'.format(epoch+1, epoch_loss / train.__len__()))
        # end = time.time()
        # print("time={}s".format(end-start))
        scheduler.step()
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
        torch.save(net.state_dict(),
                   "/home/cyf/Dataset/Task04_Hippocampus/imagesTr/CP/" + 'CP{}.pth'.format(epoch + 1))




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.02,
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
    logging.basicConfig(filename=os.path.join(os.getcwd(), 'nii' +'.txt'),
                        level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    device = args.device
    net_choice = args.net
    print(net_choice == 'ResUnet')
    # if(net_choice == 'UNet'):
    #     net = U_Net()
    # elif(net_choice == 'FCN'):
    #     net = FCN8s(n_class=1)
    # elif(net_choice == 'DeepLab'):
    #     net = DeepLab(num_classes=1,
    #                   backbone='resnet',
    #                   output_stride=16)
    # elif(net_choice == 'ResUnet'):
    #     net = R2U_Net()
    # net = UNet(n_classes=1, n_channels=3)
    net = R_ResUNet()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda(device)
        cudnn.benchmark = False  # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




