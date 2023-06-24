import numpy as np
import torch
from network import Network
from dataloader import AugDataset, ImageLoader, ImageDataset, Cropper
from torchvision.transforms import  RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, Compose, ToTensor, RandomResizedCrop, RandomErasing, Lambda
from matplotlib import pyplot as plt
import os
from skimage import metrics

np.random.seed(0)

cuda_id = 2
torch.cuda.set_device(cuda_id)

dataset_dir = './data/wsi/ah/'

model_name = 'SDN'
model_dir = dataset_dir + 'models/'
output_dir = os.path.join('./output', model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_dir = dataset_dir + 'training/'
train_list = [f for f in os.listdir(train_dir) if 'tiff' in f and f.startswith('H')]
spatial_transform = Compose([RandomResizedCrop(256), RandomErasing(value=1), RandomVerticalFlip(0.5), RandomHorizontalFlip(0.5), Lambda(lambda x: torch.permute(x, (0, 2, 1)).contiguous() if torch.rand(1).item() < 0.5 else x)])
color_transform = ColorJitter(0.5, 0.5, 0.5, 0.5)
other_transform = ToTensor()

dataset_train = AugDataset(dataset_dir + 'training/', train_list, spatial_transform, color_transform, other_transform, repeat=None)
n_batch = 8
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=n_batch, num_workers=0, shuffle=True)
n_epoch = 1000

imageLoader = ImageLoader(['H', 'A'])
dataset_test = ImageDataset(dataset_dir + 'training/', imageLoader, pair_aligned=True, transform_list=[Cropper(crop_rate=99)])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=4)
validata_list = [dataset_test[i] for i in range(4)]

model = Network(32, 32)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
mse = torch.nn.MSELoss(reduction='mean')


def cos_dis(x, y):
    return torch.mean(model.cos_dis(x, y))


for it_epoch in range(n_epoch):
    loss_epoch = 0
    model.train()
    for it_step, sample in enumerate(dataloader_train):
        optimizer.zero_grad()

        x1, x2, x11, x12, x21, x22 = [x.cuda() for x in sample]

        x_list = [[x11, x12], [x21, x22]]
        n_x = len(sample) - 2

        rec_loss = 0
        trans_loss = 0

        n_sp = 0
        n_cl = 0
        n_rec = 0
        n_trans = 0

        for i in range(n_x):
            r_i, c_i = i // 2, i % 2
            x_i = x_list[r_i][c_i]
            s_i, v_i, out_i = model.forward(x_i)
            rec_loss += mse(out_i, x_i)
            rec_loss += cos_dis(out_i, x_i)
            n_rec += 1
            for j in range(i+1, n_x):
                r_j, c_j = j // 2, j % 2
                x_j = x_list[r_j][c_j]
                s_j, v_j, out_j = model.forward(x_j)

                i2j = model.fushion_sv(s_j, v_i)
                j2i = model.fushion_sv(s_i, v_j)

                trans_loss += 0.5 * (mse(i2j, x_list[r_i][c_j]) + mse(j2i, x_list[r_j][c_i]))
                trans_loss += 0.5 * (cos_dis(i2j, x_list[r_i][c_j]) + cos_dis(j2i, x_list[r_j][c_i]))
                n_trans += 1

        loss = 100 * (rec_loss + trans_loss)
        loss.backward()
        optimizer.step()
        loss_epoch += loss
        print('Epoch %d | Iteration %d | Loss tot: %.5f, rec: %.5f, trans: %.5f' % (it_epoch, it_step, loss, rec_loss, trans_loss), end='\r', flush=True)

    mean_loss = loss_epoch / len(dataloader_train)
    print('\nEpoch %d | Loss %f' % (it_epoch, mean_loss), flush=True)
    torch.save(model.state_dict(), model_dir + model_name + '%d_%d.pth' % (cuda_id, it_epoch % 100))

    model.eval()
    psnr = []
    ssim = []
    with torch.no_grad() as ngard:
        for i, sample in enumerate(validata_list):
            sample = torch.from_numpy(sample)
            h, w = np.shape(sample)[-2:]
            sample_split = torch.split(sample.cuda(), 1, dim=0)
            x_img, y_img = sample_split
            x_img = x_img.view(-1, 3, h, w)
            y_img = y_img.view(-1, 3, h, w)

            x_s, x_v, x_out, y_s, y_v, y_out, x2y_out, y2x_out = model.transfer_full(x_img.cuda(), y_img.cuda())
            s_mse = mse(x_s, y_s)
            v_mse = mse(x_v, y_v)
            x_rec = mse(x_out, x_img)
            y_rec = mse(y_out, y_img)
            x2y_rec = mse(x2y_out, y_img)
            y2x_rec = mse(y2x_out, x_img)

            psnr.append(metrics.peak_signal_noise_ratio(x_img.cpu().numpy(), y2x_out.cpu().numpy()))
            ssim.append(metrics.structural_similarity(x_img.squeeze().permute(1, 2, 0).cpu().numpy(), y2x_out.squeeze().permute(1, 2, 0).cpu().numpy(), multichannel=True))

            print('Image %d/%d' % (i+1, len(dataloader_test)), end='\r', flush=True)

            img_list = [x_img, y_img, x_out, y_out, x2y_out, y2x_out]
            mse_list = [0, 0, x_rec, y_rec, x2y_rec, y2x_rec]
            title_list = ['x', 'y', 'x_', 'y_', 'x2y', 'y2x', 'x_v', 'y_v']
            img_list = [img.view(-1, h, w).permute(1, 2, 0).contiguous().cpu().numpy() for img in img_list]

            plt.figure()
            for j, img in enumerate(img_list):
                plt.subplot(len(img_list) // 2, 2, j+1)
                plt.xticks([])
                plt.yticks([])
                if j > 1 and j < 6:
                    plt.title(title_list[j] + '_%f' % mse_list[j])
                else:
                    plt.title(title_list[j])
                if img.shape[-1] == 1:
                    img = img.reshape(h, w)
                    plt.imshow(img, cmap='rainbow')
                else:
                    plt.imshow(img)
            plt.savefig(os.path.join(output_dir, 'vali_%d_%d.png' % (cuda_id, i)))
            plt.close()

            if i >= 3:
                break
        print("PSNR: %.5f+_%.5f" % (np.mean(psnr), np.std(psnr)))
        print("SSIM: %.5f+_%.5f" % (np.mean(ssim), np.std(ssim)))
