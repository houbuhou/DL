import torch
import torchvision.transforms.functional as tf
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
from optparse import OptionParser
from Epi.EPI_trainer import EPI
import os
from tqdm import tqdm
from unet.unet_parts import ASPPBottleneck
from sklearn import metrics


def GetArgs():
    parser = OptionParser()
    parser.add_option('-c', "--load", dest='load',
                      default=".CP34.pth", help="loaded model")
    parser.add_option('-i', "--input", dest='input',
                      default="D:" + os.sep + "epi" + os.sep + "test" + os.sep,
                      help="the dictionary of inputs")
    (options, args) = parser.parse_args()
    return options


def predict(net, image):
    net.eval()
    PredictedImage = net(image)
    PredictedImage = PredictedImage.cpu().squeeze()
    PredictedImage = tf.to_pil_image(PredictedImage)
    return PredictedImage


if __name__ == "__main__":
    args = GetArgs()
    # network = UNet(n_channels=3, n_classes=1).cuda()
    network = ASPPBottleneck(in_channels=3, out_channels=1, basic_channels=64).cuda()
    network.load_state_dict(torch.load(args.load))
    print("Model loaded from {}".format(args.load))
    MasksPath = "D:" + os.sep + "epi" + os.sep + "masks" + os.sep
    TestSet = EPI(image_paths=args.input, target_paths=MasksPath)
    for i in tqdm(range(len(TestSet))):
        image, TrueMask = TestSet[i]
        image_gpu = image.unsqueeze(0).cuda()
        PredictedImage = predict(network, image_gpu)
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.imshow(PredictedImage)
        plt.subplot(3, 1, 2)
        plt.imshow(tf.to_pil_image(image))
        plt.subplot(3, 1, 3)
        plt.imshow(tf.to_pil_image(TrueMask))
        plt.show()
