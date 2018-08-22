import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SegmentationTrainer():
    def __init__(self, checkpoint_path, epoch, batch_size, is_cuda=True):
        self.sigmoid_output = False
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = model.MySegmentator(sigmoid_output=self.sigmoid_output)
        if self.sigmoid_output:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        # RMSprop(self.model.parameters(), lr=1e-3, alpha=0.9)
        self.is_cuda = is_cuda

    def get_next_batch(self, data, masks, size):
        indicies = np.random.choice(self.get_data_len(data), size)
        return data[indicies], masks[indicies]

    def save(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def resume(self):
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        except Exception as ex:
            print('no saved model: ', ex)

    def get_data_len(self, data):
        return len(data)

    def calculate_loss(self, ground_truth, predictions):
        lambd = 1e-4
        eps = 1e-8
        loss = (1 - lambd) * self.loss(predictions, ground_truth)
        if not self.sigmoid_output:
            predictions = F.sigmoid(predictions)

        count_pred = predictions
        count_truth = ground_truth
        count_intersection = predictions * ground_truth
        for i in range(3):
            count_pred = torch.mean(count_pred, dim=1)
            count_truth = torch.mean(count_truth, dim=1)
            count_intersection = torch.mean(count_intersection, dim=1)
        dice = 2 * count_intersection / (count_pred + count_truth + eps)
        additional_loss = lambd * torch.mean(1 - dice)
        return loss + additional_loss

    def preprocess(self, images, masks):
        pass

    def train(self, train_data, train_masks, test_data, test_masks):
        self.resume()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3, alpha=0.9)
        data_size = self.get_data_len(train_data)
        iterations = int(data_size / self.batch_size)
        if self.is_cuda:
            self.model = self.model.cuda()
        for i in range(self.epoch):
            for j in range(iterations):
                images, masks = self.get_next_batch(train_data, train_masks, self.batch_size)
                if self.is_cuda:
                    images, masks = images.cuda(), masks.cuda()
                predictions = self.model(images)
                if j % 10 == 0:
                    torchvision.utils.save_image(images, './train/image.png', 5)
                    torchvision.utils.save_image(masks, './train/masks.png', 5)
                    torchvision.utils.save_image(predictions, './train/predictions.png', 5)
                    torchvision.utils.save_image(predictions * images, './train/segment.png', 5)
                loss = self.calculate_loss(masks, predictions)
                print(loss.data)
                loss.backward()
                self.optimizer.step()
            if i > 0:
                with torch.no_grad():
                    self.save()
                    test_size = int(self.get_data_len(test_data) / self.batch_size) - 1
                    cum_loss = 0
                    for j in range(test_size):
                        images, masks = test_data[j * self.batch_size:(j + 1) * self.batch_size], test_masks[
                                                                                                  j * self.batch_size:(
                                                                                                                          j + 1) * self.batch_size]
                        if self.is_cuda:
                            images, masks = images.cuda(), masks.cuda()
                        predictions = self.model(images)
                        if j == 0:
                            torchvision.utils.save_image(predictions * images, './test/segment.png', 5)
                            torchvision.utils.save_image(images, './test/image.png', 5)
                            torchvision.utils.save_image(masks, './test/mask.png', 5)
                            torchvision.utils.save_image(predictions, './test/prediction.png', 5)
                        cum_loss += self.calculate_loss(masks, predictions)
                    cum_loss /= test_size
                    print('Epoch %d\n loss: %f' % (i, cum_loss.data))
