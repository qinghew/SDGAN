import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import torch

def zero_padding(img, size0, pad1, pad2):
    zero_padding_image = np.zeros((img.shape[0], size0, size0), dtype=np.float32) #+ 255
    pad1 = pad1 / 2
    pad2 = pad2 / 2
    # print(pad1,size0 - pad1, pad2, size0 - pad2)
    zero_padding_image[:, int(pad1):int(size0 - pad1), int(pad2):int(size0 - pad2)] = img[:,:,:]
    return zero_padding_image

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        a = Image.open(A_path).convert('RGB')
        b = Image.open(B_path).convert('RGB')

        a = a.resize((200, 250), Image.BICUBIC)
        b = b.resize((200, 250), Image.BICUBIC)      
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # print(a.shape)  # torch.Size([3, 250, 200])
        a = zero_padding(a, 286, 286 - a.shape[1], 286 - a.shape[2])
        b = zero_padding(b, 286, 286 - b.shape[1], 286 - b.shape[2])


        if self.opt.phase == "train":

            a = torch.from_numpy(a) 
            b = torch.from_numpy(b)

            w_offset = random.randint(0, max(0, 286 - 256 - 1))
            h_offset = random.randint(0, max(0, 286 - 256 - 1))

            a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]


            if random.random() < 0.5:
                # print(a.shape)
                idx = [i for i in range(a.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                a = a.index_select(2, idx)
                b = b.index_select(2, idx)

        elif self.opt.phase == "test":
            a = a[:, 15:15 + 256, 15:15 + 256]
            b = b[:, 15:15 + 256, 15:15 + 256]
            a = torch.from_numpy(a) 
            b = torch.from_numpy(b)



        return {'A': a, 'B': b, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
