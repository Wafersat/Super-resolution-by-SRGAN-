# -*- coding: utf-8 -*-
# @Time    : 2022/7/17 19:12
# @File    : test.py.py
# @Software: PyCharm
import os
import argparse
from dataset import *
from models import *
import torch.nn.functional
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

# some parameter which test model
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="DIV2K_valid_HR", help="name of the test dataset")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--scale_factor", type=int, default=4, help="High-Resolution image downsampling multiples")
opt = parser.parse_args()

# creat file to save val results
os.makedirs("results", exist_ok=True)
hr_shape = (opt.img_height, opt.img_width)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, hr_shape=hr_shape, scale_factor=opt.scale_factor),
    batch_size=opt.batch_size,
    shuffle=False,
)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
Generator = GeneratorResNet()

if cuda:
    Generator.cuda()

# load model weights file which you want to test
Generator.load_state_dict(torch.load("saved_models/generator_{}.pth".format(opt.epoch)))
Generator.eval()

# test model
for index, imgs in enumerate(val_dataloader):
    lr_imgs = Variable(imgs["lr"].type(Tensor))
    hr_imgs = Variable(imgs["hr"].type(Tensor))
    gen_hr_imgs = Generator(lr_imgs)
    lr_imgs = nn.functional.interpolate(lr_imgs, scale_factor=opt.scale_factor)
    lr_imgs = make_grid(lr_imgs, nrow=1, normalize=True)
    hr_imgs = make_grid(hr_imgs, nrow=1, normalize=True)
    gen_hr = make_grid(gen_hr_imgs, nrow=1, normalize=True)
    img_grid = torch.cat((lr_imgs, gen_hr, hr_imgs), 2)
    save_image(img_grid, "results/%d.png" % (index + 1), normalize=False)
    print('The {} picture done!'.format(index + 1))

print("All picture done!")