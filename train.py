import argparse
import os
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import Discriminator, Generator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--img_channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image samples')
    parser.add_argument('--download_path', help='Path for downloaded dataset', required=True)
    parser.add_argument('--output_path', help='Path to save generated images')
    args = parser.parse_args()
    return args


def download_dataset(download_path, batch_size, num_worker=8):
    """
    Download the dataset and normalize
    :param download_path: The pathname to store the downloaded dataset.
    :param batch_size: The size of batch which is for mini-batch training
    :return: a PyTorch DataLoader of the dataset.
    """
    os.makedirs(download_path, exist_ok=True)
    # Define transform methods
    transform_methods = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Create DataLoader
    data_loader = DataLoader(datasets.FashionMNIST(download_path, download=True, transform=transform_methods),
                             batch_size=batch_size,
                             num_workers=num_worker,
                             shuffle=True)
    return data_loader


def train(data_loader, discriminator, generator, base_loss, params):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.b1, params.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=params.lr, betas=(params.b1, params.b2))
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for epoch in range(params.n_epochs):
        for i, (imgs, _) in enumerate(data_loader):

            # Adversarial ground truths
            valid = Variable(tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(tensor))

            #  Train Generator
            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(tensor(np.random.normal(0, 1, (imgs.shape[0], params.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = base_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_g.step()
            #  Train Discriminator
            optimizer_d.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = base_loss(discriminator(real_imgs), valid)
            fake_loss = base_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, params.n_epochs, i, len(data_loader),
                                                                           d_loss.item(), g_loss.item()))

            batches_done = epoch * len(data_loader) + i
            os.makedirs(params.output_path, exist_ok=True)
            if batches_done % params.sample_interval == 0:
                save_image(gen_imgs.data[:25], '%s/%d.png' % (params.output_path, batches_done), nrow=5, normalize=True)


def main():
    args = parse_arguments()
    data_loader = download_dataset(args.download_path, args.batch_size)
    img_shape = (args.img_channels, args.img_size, args.img_size)
    base_loss = nn.BCELoss()
    generator = Generator(args.latent_dim, img_shape)
    discriminator = Discriminator(img_shape)
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        base_loss.cuda()
    # Define optimizer
    train(data_loader, discriminator, generator, base_loss, args)


if __name__ == '__main__':
    main()
