import os
import numpy as np
import cPickle as pickle
from datasets.dataset import Dataset


class Cifar10(Dataset):
    """Implements the Dataset class to handle CIFAR-10.

    Attributes:
        y_dim: The dimension of label vectors (number of classes).
        split_data: A dictionary of
            {
                'train': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'val': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'test': Images of np.ndarray, Int array of labels, and int
                array of ids.
            }
    """

    def __init__(self, root='./data'):
        super(Cifar10, self).__init__('cifar10', root)
        self.y_dim = 10
        self.split_data = {}

    def load(self, split='train', lazy=True, randomize=True):
        """Implements the load function.

        Args:
            split: Dataset split, can be [train|val|test], default: train.

        Returns:
             Images of np.ndarray, Int array of labels, and int array of ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """
        if split in self.split_data.keys():
            return self.split_data[split]

        images = None
        labels = None
        data_dir = self.data_dir
        for i in range(5):
            f = open(os.path.join(data_dir, 'cifar-10-batches-py', 'data_batch_' + str(i + 1)), 'rb')
            datadict = pickle.load(f)
            f.close()

            x = datadict['data']
            y = datadict['labels']

            x = x.reshape([-1, 3, 32, 32])
            x = x.transpose([0, 2, 3, 1])

            if images is None:
                images = x
                labels = y
            else:
                images = np.concatenate((images, x), axis=0)
                labels = np.concatenate((labels, y), axis=0)

        f = open(os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch'), 'rb')
        datadict = pickle.load(f)
        f.close()

        test_images = datadict['data']
        test_labels = datadict['labels']

        test_images = test_images.reshape([-1, 3, 32, 32])
        test_images = test_images.transpose([0, 2, 3, 1])

        if split == 'train':
            images = images[:50000]
            labels = labels[:50000]
        elif split == 'val':
            images = images[40000:50000]
            labels = labels[40000:50000]
        elif split == 'test':
            images = test_images
            labels = test_labels

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)

        self.split_data[split] = [images, labels]
        self.images = images
        self.labels = labels

        return images, labels
