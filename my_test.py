import train
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tr
import model
import torchvision
import torch.nn.functional as F

def divide_data(all_data, all_masks):
    p = 0.05
    test_len = int(len(all_data) * p)
    return all_data[test_len:], all_masks[test_len:], all_data[:test_len], all_masks[:test_len]


if __name__ == '__main__':
    data = ImageFolder(root='./test_images', transform=tr.Compose([tr.Resize((1024//4,512//4)), ToTensor()]))
    all_data = []
    for img, _ in data:
        all_data.append(img)


    inference = model.UNet(is_sigmoid=False).cuda()
    inference.load_state_dict(torch.load('./train/model.ckpt'))

    train_data = torch.stack(all_data).type(torch.FloatTensor).cuda()

    example = F.hardtanh(inference(train_data),min_val=0)
    torchvision.utils.save_image(example * train_data, './train/segment.png', 5)
