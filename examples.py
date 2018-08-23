import train
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tr


def divide_data(all_data, all_masks):
    p = 0.05
    test_len = int(len(all_data) * p)
    return all_data[test_len:], all_masks[test_len:], all_data[:test_len], all_masks[:test_len]


if __name__ == '__main__':
    data = ImageFolder(root='./dataset/images', transform=tr.Compose([tr.Resize((256, 256)), ToTensor()]))
    masks = ImageFolder(root='./dataset/masks',
                        transform=tr.Compose([tr.Resize((256, 256)), tr.Grayscale(), ToTensor()]))
    all_data = []
    all_masks = []
    for img, _ in data:
        all_data.append(img)
    for img, _ in masks:
        all_masks.append(img)

    trainer = train.SegmentationTrainer('./train/model.ckpt', 1000, 2, True)
    train_data, train_mask, test_data, test_mask = divide_data(all_data, all_masks)
    train_data = torch.stack(train_data).type(torch.FloatTensor)
    train_mask = torch.stack(train_mask).type(torch.FloatTensor)
    test_data = torch.stack(test_data).type(torch.FloatTensor)
    test_mask = torch.stack(test_mask).type(torch.FloatTensor)
    print(train_data.requires_grad, train_mask.requires_grad, test_data.requires_grad, test_mask.requires_grad)
    trainer.train(train_data, train_mask, test_data, test_mask)
