import torch.utils.data as data
import torch

class RayDataset(data.Dataset):
    def __init__(self, ray_data):
        super(RayDataset, self).__init__()

        self.rayData = ray_data
        self.length = ray_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.Tensor(self.rayData[index]);


class DepthDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]





