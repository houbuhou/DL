import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data

from PIL import Image
from network import U_Net, R2AttU_Net, R2U_Net
import matplotlib.pyplot as plt
import scipy


from unet import UNet, R_ResUNet
import random
from tqdm import tqdm
from sklearn import metrics
from scipy.spatial import distance


from torchvision import transforms
import warnings
import logging
from model import FCN8s
import time
from utils import dense_crf
from modeling.deeplab import *

logging.basicConfig(filename=os.path.join(os.getcwd(), 'out.txt'),
                    level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
warnings.filterwarnings('ignore')

device = 2
hou = 0


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

        return 1-obj_dice


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


def predict_img(net,
                full_img):
    global hou
    net.eval()
    full_img = full_img.unsqueeze(0)
    start = time.time()
    pred_mask = net(full_img)
    end = time.time()
    # print(start-end)
    hou = hou + end - start
    pred_mask = pred_mask > 0.5
    pred_mask = pred_mask.float()
    # pred_mask_crop = F.sigmoid(pred_mask_crop)
    mask_pred_show = pred_mask.detach().cpu().numpy()
    mask_pred_show = mask_pred_show.squeeze().astype(np.float32)
    # plt.figure(0)
    # plt.imshow(mask_pred_show)
    # plt.figure(1)
    # img_show = img_crop.squeeze().cpu()
    # plt.imshow(img_show[0])
    # plt.show()

    pred_mask = pred_mask.detach().squeeze()
    pred_mask = pred_mask.unsqueeze(0).cpu()
    # print(pred_mask_crop.shape)

    full_mask = pred_mask.squeeze().cpu()
    full_mask = np.array(full_mask * 255, dtype=np.uint8)
    crf_img = full_img.squeeze().cpu()
    crf_img = TF.to_pil_image(crf_img)
    full_mask = dense_crf(np.array(crf_img, dtype=np.uint8), full_mask)
    full_mask = full_mask.astype(np.float32)
    full_mask = torch.from_numpy(full_mask)
    full_mask = full_mask.unsqueeze(0)
    return full_mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./checkpoints/CP99.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=False,
    #                     default=['2017-06-10_13.26.55.ndpi.16.12476_11900.2048x2048.tiff'])

    return parser.parse_args()


class MyDataset(data.Dataset):
    def __init__(self, image_paths, target_paths, resize=(512, 512),  train=True):
        self.train = train
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.files = os.listdir(self.image_paths)
        self.file_name = (f[:-5] for f in self.files)
        self.file_name = list(self.file_name)
        self.labels = os.listdir(self.target_paths)
        self.resize = resize

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=self.resize)
        image = resize(image)
        mask = resize(mask)


        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths + self.files[index])
        mask = Image.open(self.target_paths + self.file_name[index] + '._mask.jpg')
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    args = get_args()

    # net = R2AttU_Net()
    # net = UNet(n_classes=1, n_channels=3)
    # net = FCN8s(n_class=1)
    # net = DeepLab(num_classes=1,
    #               backbone='resnet',
    #               output_stride=16)
    # net = R2U_Net()
    net = R_ResUNet()
    # net = U_Net()
    dir_img = 'data/train/'
    dir_val = 'data/val/'
    dir_test = 'data/test/'
    dir_vis = 'visual/'
    dir_mask = 'data/train_masks/'

    net.cuda(device)
    net.load_state_dict(torch.load(args.model))

    test = MyDataset(image_paths=dir_test, target_paths=dir_mask, train=False)
    print("Model loaded !")
    sum_dice = 0
    sum_f1 = 0
    sum_haus = 0
    sum_time = 0

    for i in tqdm(range(test.__len__())):
        img, true_masks = test[i]
        img = img.cuda(device)
        start = time.time()
        pred_masks = predict_img(net=net,
                           full_img=img)
        end = time.time()
        sum_time = sum_time + end - start
        img_show = img.cpu().squeeze()
        img_show = TF.to_pil_image(img_show)
        mask_pred_show = pred_masks.detach().cpu().numpy()
        mask_pred_show = mask_pred_show.squeeze().astype(np.float32)
        true_masks_show = true_masks.squeeze()
        dice = dice_coeff(pred_masks, true_masks)
        haus = distance.directed_hausdorff(pred_masks.squeeze().numpy(), true_masks.squeeze().numpy())[0]
        true_masks_flat = true_masks.view(-1).numpy().astype(np.uint8)
        pred_masks_flat = pred_masks.view(-1).numpy().astype(np.uint8)
        f1 = metrics.f1_score(true_masks_flat, pred_masks_flat, pos_label=0)

        if dice > 1:
            fig = plt.figure(0)
            a = fig.add_subplot(1, 3, 1)
            a.set_title('image')
            plt.imshow(img_show)

            b = fig.add_subplot(1, 3, 2)
            b.set_title('true mask')
            plt.imshow(true_masks_show)

            c = fig.add_subplot(1, 3, 3)
            c.set_title('predict')
            plt.imshow(mask_pred_show)
            # plt.imsave(str(i)+'out.png', mask_pred_show)
            scipy.misc.imsave(str(i)+'out.png', mask_pred_show)

            plt.show()
        print(dice)
        print(f1)
        print(haus)
        sum_dice += dice
        sum_f1 += f1
        sum_haus += haus
        logging.info('Dice={}'.format(dice))

    print(sum_dice / test.__len__())
    print(sum_f1 / test.__len__())
    print(sum_haus / test.__len__())
    print(sum_time / (test.__len__()))
    print(hou/(test.__len__()))


