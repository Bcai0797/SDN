import os
import torch
import numpy as np
from PIL import Image
from functools import reduce
from skimage.transform import resize


class Resizer(object):
    def __init__(self, h=800, w=1300):
        self.h = h
        self.w = w

    def __call__(self, images):
        h = self.h
        w = self.w
        images = np.array([resize(img, (h, w), preserve_range=True) for img in images])
        return images
    

class Cropper(object):
    def __init__(self, crop_h=256, crop_w=256, crop_rate=0.9, seed=0):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.crop_rate = crop_rate
        self.rs = np.random.RandomState(seed)

    def __call__(self, images):
        h, w, c = np.shape(images)[1:]
        th, tw = self.rs.randint(h - self.crop_h), self.rs.randint(w - self.crop_w)
        c_rate = self.rs.rand()
        if h < self.crop_h or w < self.crop_w or c_rate >= self.crop_rate:
            return Resizer(self.crop_h, self.crop_w)(images)
        ret = []
        for i in range(len(images)):
            img = images[i]
            tx = 0 if h == self.crop_h else th
            ty = 0 if w == self.crop_w else tw
            img = img[tx:(tx+self.crop_h), ty:(ty+self.crop_w), :]
            ret.append(img)
        return np.array(ret)


class ImageLoader(object):
    def __init__(self, pre_list, shuffle=False, level_idx=None):
        self.pre_list = pre_list
        self.shuffle = shuffle
        self.level = np.array([['#', 'A', 'B', 'C', 'D'], ['#', 'a', 'b', 'c', 'd']])
        if level_idx is not None:
            self.level = [self.level[level_idx][1:]]

    def __call__(self, image_name):
        if len(self.pre_list) > 1:
            return [p + image_name for p in self.pre_list]
        else:
            len_name = len(image_name)
            assert len_name >= 5 and len_name <= 7

            level_idx = [np.random.choice(le) for le in self.level]
            o_name = self.pre_list[0] + image_name
            t_name = self.pre_list[0] + image_name[:5] + level_idx[0] + level_idx[1]
            t_name = t_name.split('#')[0]
            ret = np.array([o_name, t_name])
            if self.shuffle:
                np.random.shuffle(ret)
            return ret


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, img_loader=None, pair_aligned=False, transform_list=None, repeat=None, shuffle=None, transform_style=False):
        self.dataset_dir = dataset_dir
        self.img_loader = img_loader
        self.pair_aligned = pair_aligned
        self.transform_list = transform_list
        self.repeat = repeat

        self.image_names = [f[:-5] for f in os.listdir(dataset_dir) if 'tiff' in f]
        if img_loader is not None:
            self.image_names = [f[1:] for f in self.image_names if 'H' in f]
        else:
            self.img_loader = lambda x: [x]

        if repeat is not None and repeat > 1:
            repeat = int(np.ceil(repeat))
            self.image_names = np.repeat(self.image_names, repeat)
        if shuffle:
            np.random.shuffle(self.image_names)

        self.transform_func = self.transform_img_style if transform_style else self.transform_img

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        sample = self.img_loader(self.image_names[idx])
        if self.pair_aligned:
            ret = self.transform_func(self.load_img(sample))
        else:
            ret = self.transform_func(self.load_img(sample))
            rh, rw = np.shape(ret)[-2:]
            ref = self.load_img(sample[0:1])[0]
            h, w, _ = np.shape(ref)
            h, w = h // 2, w // 2
            rh, rw = rh // 2, rw // 2
            sh, th, sw, tw = h - rh, h + rh, w - rw, w + rw
            ref = np.transpose(np.array([ref[sh:th, sw:tw, :]]), (0, 3, 1, 2)) / 255
            ret = np.concatenate([ret, ref], axis=0)
        return ret

    def load_img(self, img_name_list):
        img = np.asarray([np.array(Image.open(self.dataset_dir + img_name + '.tiff')) for img_name in img_name_list], dtype=np.float32)
        return img

    def transform_img(self, imgs):
        ret = imgs
        if self.transform_list is not None:
            ret = reduce(lambda acc, val: val(acc), self.transform_list, imgs)
        return np.transpose(ret, (0, 3, 1, 2)) / 255


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, img_list, spatial_transform, color_transform, other_transform, repeat=None, mode='train', seed=0):
        self.dataset_dir = dataset_dir
        self.img_list = img_list
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.other_transform = other_transform
        self.repeat = repeat
        self.len = len(self.img_list)
        self.mode = mode
        if repeat is not None and repeat > 1:
            repeat = int(np.ceil(repeat))
            self.len = repeat * len(self.img_list)
        self.rs = np.random.RandomState(seed)
        if mode != 'train':
            self.rd_idx = np.arange(self.len)
            self.rs.shuffle(self.rd_idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.mode == 'train':
            idx = idx % self.len
            img = self.__getimg__(idx)

            x = self.other_transform(img)

            x1 = self.spatial_transform(x)
            x2 = self.spatial_transform(x)

            # t1 = self.color_transform.get_params([0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [-0.5, 0.5])
            # t2 = self.color_transform.get_params([0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [-0.5, 0.5])

            # x11, x21 = t1(x1), t1(x2)
            # x12, x22 = t2(x1), t2(x2)
            x11, x21 = self.color_transform(torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)]))
            x12, x22 = self.color_transform(torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)]))
            
            # x1, x2, x11, x12, x21, x22 = [self.other_transform(i) for i in [x1, x2, x11, x12, x21, x22]]

            return x1, x2, x11, x12, x21, x22
        else:
            idx_1 = idx % self.len
            idx_2 = self.rd_idx[idx_1]

            x1 = self.__getimg__(idx_1)
            x2 = self.__getimg__(idx_2)

            x1, x2 = [self.other_transform(i) for i in [x1, x2]]
            return x1, x2

    def __getimg__(self, idx):
        imgpath = self.dataset_dir + self.img_list[idx]
        img = Image.open(imgpath)
        return img


class ImageClassDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, center_list, pre_transform, aug_transform, mode='train'):
        img_list = np.array([datadir + 'center_' + str(cid) + '/' + f for cid in center_list for f in os.listdir(datadir + 'center_' + str(cid) + '/') if f.endswith('bmp')])
        img_list = np.sort(img_list)
        label_list = np.asarray([0 if f.split('/')[-1].startswith('nt_') else 1 for f in img_list], dtype=np.float32)
        weight_list = np.ones(np.shape(label_list), dtype=np.float32)

        self.tot = len(label_list)
        self.n_t = int(np.sum(label_list))
        self.n_nt = self.tot - self.n_t
        self.pre_transform = pre_transform
        self.aug_transform = aug_transform
        self.mode = mode
        self.img_list = img_list
        self.label_list = label_list
        if len(weight_list) > 0:
            self.weight_list = weight_list * (label_list * self.n_nt + (1-label_list) * self.n_t)
            self.weight_list /= np.sum(self.weight_list)

    def __len__(self):
        return self.tot

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        img = self.__getimg__(idx)
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        label = self.label_list[idx]
        if self.mode == 'train' and self.aug_transform is not None:
            img = self.aug_transform(img)
        return img, label

    def __getimg__(self, idx):
        imgpath = self.img_list[idx]
        return Image.open(imgpath)
    
    def getWeight(self):
        return self.weight_list


class ImageTransDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, pre_transform):
        self.img_list = img_list
        img_list = np.sort(img_list)
        # print(img_list[:10])
        label_list = np.asarray([0 if 'nt_' in f.split('/')[-2].replace('patient', '') else 1 for f in img_list], dtype=np.float)
        weight_list = np.ones(np.shape(label_list), dtype=np.float)

        self.tot = len(label_list)
        self.n_t = int(np.sum(label_list))
        self.n_nt = self.tot - self.n_t
        self.pre_transform = pre_transform
        self.img_list = img_list
        self.label_list = label_list
        self.weight_list = weight_list * (label_list * self.n_nt + (1-label_list) * self.n_t)
        # print(self.tot, self.n_t, self.n_nt)
        self.weight_list /= np.sum(self.weight_list)

    def __len__(self):
        return self.tot

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        img = self.__getimg__(idx)
        img = self.pre_transform(img)
        label = self.label_list[idx]
        return img, label

    def __getimg__(self, idx):
        imgpath = self.img_list[idx]
        return Image.open(imgpath)
    
    def getWeight(self):
        return self.weight_list
