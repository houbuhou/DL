import torch
import torchvision.transforms.functional as tf
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet.unet_model import DepthUNet, UNet, CNN, NLFFUNet
from optparse import OptionParser
from Epi.EPI_trainer import EPI
import os
from tqdm import tqdm
from unet.unet_parts import ASPPBottleneck
from sklearn import metrics
import torch.nn.functional as F


def FeatureVisualization(module, input):
    x = input[0][0]
    min_num = np.minimum(64, x.size()[0])
    for i in range(min_num):
        plt.subplot(8, 8, i + 1)
        plt.imshow(x[i].detach().squeeze().cpu())

    plt.show()


def GetArgs():
    parser = OptionParser()
    parser.add_option('-c', "--load", dest='load',
                      default=".CP6.pth", help="loaded model")
    parser.add_option('-i', "--input", dest='input',
                      default="D:" + os.sep + "epi" + os.sep + "test" + os.sep,
                      help="the dictionary of inputs")
    (options, args) = parser.parse_args()
    return options


def dice_coefficient(input, target):
    intersection = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + 1e-6

    dice = (2 * intersection) / union

    return dice.item()



def predict(net, image):
    net.eval()
    PredictedImage = net(image)
    PredictedImage = (PredictedImage >= 0.5) * 1.0
    # PredictedImage = F.sigmoid(PredictedImage)
    PredictedImage = PredictedImage.cpu().squeeze()
    PredictedImage = tf.to_pil_image(PredictedImage)
    return PredictedImage


def infer(net, image):
    net.eval()
    Predicted = net(image)
    # Predicted = (Predicted >= 0.5) * 1.0
    # PredictedImage = F.sigmoid(PredictedImage)

    return Predicted.cpu().item()


if __name__ == "__main__":
    args = GetArgs()
    # network = UNet(n_channels=3, n_classes=1).cuda()
    # network = ASPPBottleneck(in_channels=3, out_channels=1, basic_channels=64).cuda()
    # network = DepthUNet(n_classes=1, n_channels=3).cuda()
    # network = CNN().cuda()
    network = NLFFUNet(n_classes=1, n_channels=3).cuda()
    # for name, m in network.named_modules():
    #     print(name)
    #     if name == 'InConv.DepthConv.1':
    #         m.register_forward_pre_hook(FeatureVisualization)
    #     if isinstance(m, torch.nn.Conv2d):
    #         m.register_forward_pre_hook(FeatureVisualization)
    network.load_state_dict(torch.load(args.load))
    print("Model loaded from {}".format(args.load))
    MasksPath = "D:" + os.sep + "epi" + os.sep + "masks" + os.sep
    TestSet = EPI(image_paths=args.input, target_paths=MasksPath, train=False)
    a = 0
    for i in tqdm(range(len(TestSet))):
        image, TrueMask, label = TestSet[i]
        image_gpu = image.unsqueeze(0).cuda()
        PredictedImage = predict(network, image_gpu)
        print(dice_coefficient(TrueMask, tf.to_tensor(PredictedImage)))
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.imshow(PredictedImage)
        plt.subplot(3, 1, 2)
        plt.imshow(tf.to_pil_image(image))
        plt.subplot(3, 1, 3)
        plt.imshow(tf.to_pil_image(TrueMask))
        plt.show()
