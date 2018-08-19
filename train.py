import model
import numpy as np

class SegmentationTrainer():

    def __init__(self, checkpoint_path, epoch, batch_size, is_cuda = True):
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = model.MySegmentator()

    def get_next_batch(self, data, size):
        return np.random.choice(data,size)

    def get_data_len(self,data):
        return len(data)

    def calculate_loss(self, ground_truth, predictions):
        pass

    def train(self, data):
        data_size = self.get_data_len(data)
        iterations = int(data_size/self.batch_size)
        for i in range(self.epoch):
            for j in range(iterations):
                images, masks = self.get_next_batch(data, self.batch_size)
                predictions = model(images)
                x = self.calculate_loss(masks, predictions)




