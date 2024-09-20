import os
import torch
from torch.utils.data.distributed import DistributedSampler
from scipy.io import loadmat
import numpy as np
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV


def get_dataloader(data_dir, file_range, target_step, batch_size, train, distributed, stride=1, 
                   num_frames=1, num_out_frames=1, num_workers=1, pin_memory=True):

    dataset = TurbulenceDataset(data_dir=data_dir, file_range=file_range, target_step=target_step, 
                                stride=stride, num_frames=num_frames, num_out_frames=num_out_frames)

    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train and not distributed:
        sampler = torch.utils.data.RandomSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,  # (sampler is None)
                                             sampler=sampler,  # if train else None
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)


    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class TurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_range, target_step, stride, num_frames, num_out_frames):
        """
        Args:
            data_dir (str): Directory with .mat data files.
            file_range (tuple): Range of file numbers (start, end).
            target_step (int): Number of steps forward for target output.
            stride (int): Number of steps between samples.
            num_frames (int): Number of time samples for input.
            num_out_frames (int): Number of time samples for output.
        """
        self.data_dir = data_dir
        self.input_file_numbers = range(file_range[0] + (num_frames - 1), file_range[1] + 1, stride)
        self.label_file_numbers = range(file_range[0] + (num_frames - 1) + target_step, file_range[1]+1+target_step, stride)
        self.input_file_list = [os.path.join(data_dir, f"{i}.mat") for i in self.input_file_numbers]
        self.label_file_list = [os.path.join(data_dir, f"{i}.mat") for i in self.label_file_numbers]
        self.target_step = target_step
        self.stride = stride
        self.num_frames = num_frames
        self.num_out_frames = num_out_frames

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        """
        Args:
          idx (int): Index of the file to load.

        Returns:
          tuple (torch.Tensor): Data loaded from the .mat file.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp_tensor, label_tensor = [], []
        for t in range(self.num_frames):
            inp_data = self.get_item_input_one_step(idx - t)   # output (tensor) shape: [C=2, T=1, X, Y]
            inp_tensor.append(inp_data)
        for t in range(self.num_out_frames):
            label_data = self.get_item_label_one_step(idx - t)   # output (tensor): [C=2, T=2, X, Y]
            label_tensor.append(label_data)

        inp_tensor = torch.cat(inp_tensor, dim=1)  # along T dim
        label_tensor = torch.cat(label_tensor, dim=1)

        return (inp_tensor, label_tensor)

    def get_item_input_one_step(self, idx):

        input_file_path = self.input_file_list[idx]
        input_mat_data = loadmat(input_file_path)
        input_Omega = input_mat_data['Omega']
        input_data_tensor = self.omega2uv(input_Omega).unsqueeze(1)


        return input_data_tensor

    def get_item_label_one_step(self, idx):

        label_file_path = self.label_file_list[idx]
        label_mat_data = loadmat(label_file_path)
        label_Omega = label_mat_data['Omega']
        label_data_tensor = self.omega2uv(label_Omega).unsqueeze(1)

        return label_data_tensor

    def omega2uv(self, Omega):
        """
        Args:
          Omega (np.array): 2D Omega data.
        Returns:
          data_tensor (torch.Tensor): U, V data.
        """
        nx, ny = Omega.shape
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

        Psi = Omega2Psi(Omega, invKsq)
        U, V = Psi2UV(Psi, Kx, Ky)

        # Combine U and V into a single tensor with 2 channels
        data_tensor = torch.tensor(np.stack([U, V]), dtype=torch.float32)

        return data_tensor
