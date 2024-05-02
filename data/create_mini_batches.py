"""
Create and yield Mini Batches in different ways.
"""

import os
import numpy as np
import copy
import random

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from utils.utils import get_files_by_extension

from collections import defaultdict
from data import augmentation


def get_data_sampler(arguments, preprocess_func):
    """
    Generate Sequence with desired sampling strategy.

    Args:
        arguments: Namespace
            Contains all arguments, which where defined from command line when running the script.
            Used parameter from arguments in this function: [sampler, validation_data_dir]
        preprocess_func: function
            Function which preprocesses the data

    Returns:
        keras sequence

    """

    input_size = tuple(arguments.input_size)
    input_shape = input_size + (3,)

    if arguments.sampler == "triplet":
        train_dat_seq = TripletSequence(train_dat_dir=arguments.dataset_dir,
                                        batch_size=arguments.batch_size,
                                        preprocess=preprocess_func,
                                        input_size=input_shape,
                                        rae=arguments.random_erasing_augmentation,
                                        random_crop=arguments.random_cropping,
                                        num_instances=arguments.number_instances)
    elif arguments.sampler == "balanced_triplet":
        train_dat_seq = BalancedTripletSequence(train_dat_dir=arguments.dataset_dir,
                                                batch_size=arguments.batch_size,
                                                preprocess=preprocess_func,
                                                input_size=input_shape,
                                                rae=arguments.random_erasing_augmentation,
                                                random_crop=arguments.random_cropping,
                                                num_instances=arguments.number_instances)
    else:
        train_dat_seq = RandomSequence(train_dat_dir=arguments.dataset_dir,
                                       batch_size=arguments.batch_size,
                                       preprocess=preprocess_func,
                                       input_size=input_shape,
                                       rae=arguments.random_erasing_augmentation,
                                       random_crop=arguments.random_cropping)

    if arguments.validation_data_dir:
        val_dat_seq = ValidationDataSequence(val_dat_dir=arguments.validation_data_dir,
                                         # val_dat_dir="/datasets_nas/jowa3080/all_data_val/val_imgs",
                                         batch_size=arguments.batch_size,
                                         num_class=train_dat_seq.num_class,
                                         preprocess=preprocess_func,
                                         y_true_to_categorical=True)
        return train_dat_seq, val_dat_seq
    else:
        return train_dat_seq


class ValidationDataSequence(Sequence):
    # For more documentation see https://keras.io/utils/#sequence
    def __init__(self, val_dat_dir, batch_size, num_class,
                 preprocess, dummy_data=None, extension=('.png', '.jpg'),
                 test_mode=False, y_true_to_categorical=True):
        """
        Implements a sequence of images for validation.

        Parameters
        ----------
        val_dat_dir : str
            Path to train data directory
        batch_size  : int
            Mini batch size
        num_class : int
            Number of ids in train data
        preprocess : function
            Preprocess function that expects an file path as the input
        dummy_data : numpy.list
            If None: no dummy data will be feed to the neural network
            Else: Dummy data will be added to y_label
        extension : {str, tuple, None}
            File extensions for the files to load
        test_mode : bool
            With test_mode labels won't be returned
        y_true_to_categorical : bool
            If True: to_categorical will be called on y_true

        Returns
        -------
        keras sequence
        """
        self.batch_size = batch_size
        self.file_list = get_files_by_extension(val_dat_dir,
                                                extension=extension,
                                                recursive=True,
                                                flat_structure=True)
        self.num_class = num_class
        self.preprocess = preprocess
        self.test_mode = test_mode
        self.dummy_data = dummy_data
        self.y_true_to_categorical = y_true_to_categorical

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        file_list_batch = self.file_list[idx * self.batch_size:
                                         (idx + 1) * self.batch_size]

        if self.test_mode:
            return np.array([self.preprocess(path) for path in file_list_batch]), \
                            to_categorical(np.random.randint(0, self.num_class, len(file_list_batch)), self.num_class)

        if self.y_true_to_categorical:
            batch_ids = \
                [to_categorical(int(file_name.rsplit(os.sep, 2)[1]), self.num_class)
                 for file_name in file_list_batch]
        else:
            batch_ids = [[int(file_name.rsplit(os.sep, 2)[1])] for file_name in
                         file_list_batch]

        return np.array([self.preprocess(path)
                         for path in file_list_batch]), np.array(batch_ids)

    def __bool__(self):
        # Needed for Keras in Python 3
        return True


class TripletSequence(Sequence):
    # For more documentation see https://keras.io/utils/#sequence
    def __init__(self, train_dat_dir, batch_size, preprocess, input_size, dummy_labels=False, rae=False,
                 random_crop=False, num_instances=4, extension=('.png', '.jpg')):
        """
        Implements an identity sampling strategy, where P Persons are sampled at first and afterwards
        K images of each person are sampled. Identities with more than K images can be used in more
        then one batch per epoch to use as much as possible different data.

        Parameters
        ----------
        train_dat_dir : str
            Path to train data directory
        batch_size  : int
            Mini batch size
        preprocess : function
            Preprocess function that expects an file path as the input
        input_size : tuple
            Inputs size of images
        dummy_labels : bool
            if dummy labels should be added to the training data
        rae : bool
            if random erasing augmentation should be used
        random_crop: bool
            if random cropping should be used
        num_instances : int
            number of images per person in each batch
        extension : {str, tuple, None}
            File extensions for the files to load

        Returns
        -------
        keras sequence
        """

        self.file_list = get_files_by_extension(train_dat_dir,
                                                extension=extension,
                                                recursive=True,
                                                flat_structure=True)
        self.data_source = [(path, int(path.rsplit(os.sep, 2)[1])) for path in self.file_list]
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.input_size = input_size
        self.dummy_labels = dummy_labels
        self.rae = rae
        self.random_crop = random_crop

        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_class = len(self.pids)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __getitem__(self, index):
        batch_pids = self.final_idxs[index * self.batch_size: (index+1) * self.batch_size]

        x_data = [np.array([self.preprocess(self.data_source[i][0]) for i in batch_pids])]
        y_data = [np.array([to_categorical(self.data_source[i][1], self.num_class) for i in batch_pids])]

        x_processed = []
        for img in x_data[0]:

            # horizontal flipping
            flip = random.choice([0, 1])
            if flip:
                image = np.flip(img, axis=1)
            else:
                image = img

            # padding and random cropping
            if self.random_crop:
                padded = tf.pad(image, [[10, 10], [10, 10], [0, 0]])
                cropped = tf.image.random_crop(padded, self.input_size)
            else:
                cropped = tf.convert_to_tensor(image)

            # random erasing augmentation
            if self.rae:
                augmented = augmentation.random_erasing_augmentation(cropped)
                x_processed.append(augmented)
            else:
                x_processed.append(cropped)

        x_processed = [tf.stack(x_processed)]

        return x_processed, y_data

    def __len__(self):
        return self.length // self.batch_size

    def create_batches(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        self.final_idxs = final_idxs


class BalancedTripletSequence(Sequence):
    # For more documentation see https://keras.io/utils/#sequence
    def __init__(self, train_dat_dir, batch_size, preprocess, input_size, dummy_labels=False, rae=False,
                 random_crop=False, num_instances=4,
                 extension=('.png', '.jpg')):
        """
        Implements an identity sampling strategy, where P Persons are sampled at first and afterwards
        K images of each person are sampled. Identities with more then K images are only used once per
        epoch to enable a balanced class distribution.

        Parameters
        ----------
        train_dat_dir : str
            Path to train data directory
        batch_size  : int
            Mini batch size
        preprocess : function
            Preprocess function that expects an file path as the input
        input_size : tuple
            Inputs size of images
        dummy_labels : bool
            if dummy labels should be added to the training data
        rae : bool
            if random erasing augmentation should be used
        random_crop: bool
            if random cropping should be used
        num_instances : int
            number of images per person in each batch
        extension : {str, tuple, None}
            File extensions for the files to load

        Returns
        -------
        keras sequence
        """

        self.file_list = get_files_by_extension(train_dat_dir,
                                                extension=extension,
                                                recursive=True,
                                                flat_structure=True)
        self.data_source = [(path, int(path.rsplit(os.sep, 2)[1])) for path in self.file_list]
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.input_size = input_size
        self.dummy_labels = dummy_labels
        self.rae = rae
        self.random_crop = random_crop

        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_class = len(self.pids)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __getitem__(self, index):
        batch_pids = self.final_idxs[index * self.batch_size: (index+1) * self.batch_size]

        x_data = [np.array([self.preprocess(self.data_source[i][0]) for i in batch_pids])]
        y_data = [np.array([to_categorical(self.data_source[i][1], self.num_class) for i in batch_pids])]

        x_processed = []
        for img in x_data[0]:

            # horizontal flipping
            flip = random.choice([0, 1])
            if flip:
                image = np.flip(img, axis=1)
            else:
                image = img

            # padding and random cropping
            if self.random_crop:
                padded = tf.pad(image, [[10, 10], [10, 10], [0, 0]])
                cropped = tf.image.random_crop(padded, self.input_size)
            else:
                cropped = tf.convert_to_tensor(image)

            # random erasing augmentation
            if self.rae:
                augmented = augmentation.random_erasing_augmentation(cropped)
                x_processed.append(augmented)
            else:
                x_processed.append(cropped)

        x_processed = [tf.stack(x_processed)]

        return x_processed, y_data

    def __len__(self):
        return self.length // self.batch_size

    def create_batches(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            # for idx in idxs:
            for idx in idxs[:self.num_instances]:
                batch_idxs.append(idx)
            batch_idxs_dict[pid].append(batch_idxs)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        self.final_idxs = final_idxs


class RandomSequence(Sequence):
    # For more documentation see https://keras.io/utils/#sequence
    def __init__(self, train_dat_dir, batch_size, preprocess, input_size, dummy_labels=False, rae=False,
                 random_crop=False, extension=('.png', '.jpg')):
        """
        Implements an random sampling strategy.

        Parameters
        ----------
        train_dat_dir : str
            Path to train data directory
        batch_size  : int
            Mini batch size
        preprocess : function
            Preprocess function that expects an file path as the input
        input_size : tuple
            Inputs size of images
        dummy_labels : bool
            if dummy labels should be added to the training data
        rae : bool
            if random erasing augmentation should be used
        random_crop: bool
            if random cropping should be used
        extension : {str, tuple, None}
            File extensions for the files to load

        Returns
        -------
        keras sequence
        """

        self.file_list = get_files_by_extension(train_dat_dir,
                                                extension=extension,
                                                recursive=True,
                                                flat_structure=True)
        self.data_source = [(path, int(path.rsplit(os.sep, 2)[1])) for path in self.file_list]
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.input_size = input_size
        self.dummy_labels = dummy_labels
        self.rae = rae
        self.random_crop = random_crop

        self.index_dic = defaultdict(list)

        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_class = len(self.pids)

        self.length = len(self.file_list)
        self.final_idxs = list(range(len(self.file_list)))

    def __getitem__(self, index):

        batch_imgs = self.final_idxs[index * self.batch_size: (index+1) * self.batch_size]

        x_data = [np.array([self.preprocess(self.data_source[i][0]) for i in batch_imgs])]
        y_data = [np.array([to_categorical(self.data_source[i][1], self.num_class) for i in batch_imgs])]

        x_processed = []
        for img in x_data[0]:

            # horizontal flipping
            flip = random.choice([0, 1])
            if flip:
                image = np.flip(img, axis=1)
            else:
                image = img

            # padding and random cropping
            if self.random_crop:
                padded = tf.pad(image, [[10, 10], [10, 10], [0, 0]])
                cropped = tf.image.random_crop(padded, self.input_size)
            else:
                cropped = tf.convert_to_tensor(image)

            # random erasing augmentation
            if self.rae:
                augmented = augmentation.random_erasing_augmentation(cropped)
                x_processed.append(augmented)
            else:
                x_processed.append(cropped)

        x_processed = [tf.stack(x_processed)]

        return x_processed, y_data

    def __len__(self):
        return self.length // self.batch_size

    def create_batches(self):
        random.shuffle(self.final_idxs)
