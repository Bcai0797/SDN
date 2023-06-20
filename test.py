import numpy as np
import torch
from network import Network
from dataloader import Cropper, ImageLoader, ImageDataset
from skimage import metrics
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(0)
torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flag_save = True
dataset_dir = './data/wsi/ah/'
model_name = 'Aug_mse2'
c_size = 256
ref_size = 256
if not os.path.exists('./data/wsi/ah/%d_test_patch' % c_size):
    os.mkdir('./data/wsi/ah/%d_test_patch' % c_size)
ref_dir = './data/wsi/ah/%d_test_patch' % ref_size
model_dir = dataset_dir + 'models/'
imageLoader = ImageLoader(['H', 'A'])
dataset_test = ImageDataset('./data/wsi/ah/testing/', imageLoader, pair_aligned=False, transform_list=[Cropper(crop_rate=99, crop_h=c_size, crop_w=c_size)])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=4)
n_epoch = 100

model = Network(32, 32)

state_dict = torch.load(model_dir + model_name + '.pth', map_location=device)

model.load_state_dict(state_dict)
model.to(device)

mse = torch.nn.MSELoss(reduction='mean')

model.eval()
psnr = []
ssim = []

time_list = []

with torch.no_grad() as ngard:

    for i, sample in enumerate(dataloader_test):
        h, w = np.shape(sample)[-2:]
        sample_split = torch.split(sample.to(device), 1, dim=1)
        x_img, y_img, rf_img = sample_split
        x_img = x_img.view(-1, 3, h, w).to(device)
        y_img = y_img.view(-1, 3, h, w).to(device)
        rf_img = rf_img.view(-1, 3, h, w).to(device)
        
        y2x_out = model.transfer_x2y(y_img, rf_img)

        x_im = x_img.squeeze().permute(1, 2, 0).cpu().numpy()
        y2x_im = y2x_out.squeeze().permute(1, 2, 0).cpu().numpy()

        im1 = x_im
        im2 = y2x_im

        im_psnr = metrics.peak_signal_noise_ratio(im1, im2)
        im_ssim = metrics.structural_similarity(im1, im2, channel_axis=-1)

        psnr.append(im_psnr)
        ssim.append(im_ssim)

        print('Image %d/%d' % (i+1, len(dataloader_test)), end='\r', flush=True)

    print()
    print("PSNR: %.5f+_%.5f" % (np.mean(psnr), np.std(psnr)))
    print("SSIM: %.5f+_%.5f" % (np.mean(ssim), np.std(ssim)))
