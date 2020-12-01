'''
Authors: Geeticka Chauhan, Ruizhi Liao

This script contains data I/O and preprocessing utilities used by the model script
'''
from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import logging
from scipy.stats import logistic
import numpy as np
from skimage import io
import scipy.ndimage as ndimage
from shutil import copyfile
import shutil
import time
from zipfile import ZipFile
from math import floor, ceil

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from pytorch_transformers.modeling_bert import BertPreTrainedModel
from pytorch_transformers import BertModel

from joint_img_txt.model.convert_examples_to_features import convert_examples_to_features_multilabel
from joint_img_txt.model.convert_examples_to_features import convert_examples_to_features

csv.field_size_limit(2147483647) 
# Increase CSV reader's field limit incase we have long text.


# adapted from
# https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca
def load_and_cache_examples(args, tokenizer):
    logger = logging.getLogger(__name__)

    '''
    Load text features if they have been pre-processed;
    otherwise pre-process the raw text and save the features
    '''
    processor = EdemaMultiLabelClassificationProcessor() \
        if args.output_channel_encoding == 'multilabel' \
        else EdemaClassificationProcessor()
    num_labels = len(processor.get_labels())

    if args.output_channel_encoding == 'multilabel':
        get_features = convert_examples_to_features_multilabel
    else:
        get_features = convert_examples_to_features
    cached_features_file = os.path.join(
        args.text_data_dir,
        f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")
    if os.path.exists(cached_features_file) and not args.reprocess_input_data:
        logger.info("Loading features from cached file %s", cached_features_file)
        print("Loading features from cached file %s"%cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.text_data_dir)
        label_list = processor.get_labels()
        examples = processor.get_all_examples(args.text_data_dir)
        features = get_features(examples, label_list, args.max_seq_length, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        print("Saving features into cached file %s"%cached_features_file)
        torch.save(features, cached_features_file)

    all_txt_tokens = {f.report_id: f.input_ids for f in features}
    all_txt_masks = {f.report_id: f.input_mask for f in features}
    all_txt_segments = {f.report_id: f.segment_ids for f in features}
    all_txt_labels = {f.report_id: f.label_id for f in features}

    '''
    Split the data for training/evaluation
    '''
    if args.data_split_mode == 'testing':
        use_test_data = True
    else:
        use_test_data = False
    train_img_labels, train_img_txt_ids, val_img_labels, val_img_txt_ids = \
        _split_tr_val(args.data_split_path, 
                      args.training_folds, 
                      args.validation_folds,
                      use_test_data=use_test_data,
                      use_all_data=args.use_all_data)
    if args.do_eval:
        all_img_txt_ids = val_img_txt_ids
        all_img_labels = val_img_labels
    if args.do_train:
        all_img_txt_ids = train_img_txt_ids
        all_img_labels = train_img_labels
    print("Length of all image text ids", len(all_img_txt_ids))

    '''
    Specify the image pre-processing method 
    depending on it's for training/evaluation
    '''
    if args.do_train:
        xray_transform = RandomTranslateCrop(2048)
    if args.do_eval:
        xray_transform = CenterCrop(2048)

    '''
    Instantiate the image-text dataset
    '''
    dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, all_img_txt_ids, args.img_data_dir, 
                                  all_img_labels, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)
    print("Length of the dataset is ", len(dataset))

    return dataset, num_labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, report_id, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
            report_id: id of the report like 4345466 without s or txt extension
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.report_id = report_id


class EdemaClassificationProcessor(DataProcessor):
    """Processor for multi class classification dataset. 
    Assume reading from a multiclass file
    so the label will be in 0-3 format
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_all_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            labels = line[1]
            report_id = line[2]
            examples.append(
                InputExample(
                    report_id=report_id, guid=guid, 
                    text_a=text_a, text_b=None, labels=labels))
        return examples


# Note the multilabel classification refers to the classification for ordinally encoded labels
# In the multilabel case, we will just write multiple labels as one string in the tsv file
# So the actual DataProcessor between multi-class and multi-label classification will be the same
class EdemaMultiLabelClassificationProcessor(DataProcessor):
    """Processor for multi label classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_all_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            labels = [ char for char in line[1]]
            report_id = line[2]
            examples.append(
                InputExample(
                    report_id=report_id, guid=guid, 
                    text_a=text_a, text_b=None, labels=labels))
        return examples


# Given a data split list (.csv), training folds, validation folds, or test folds,
# return DICOM IDs and the associated labels for training and evaluation
def _split_tr_val(split_list_path, training_folds, validation_folds, 
                  use_test_data=False, use_all_data=False):
    """Extracting finding labels
    """

    print('Data split list being used: ', split_list_path)

    train_labels = {}
    train_ids = []
    train_img_txt_ids = {}
    val_labels = {}
    val_ids = []
    val_img_txt_ids = {}
    test_labels = {}
    test_ids = []
    test_img_txt_ids = {}

    with open(split_list_path, 'r') as train_label_file:
        train_label_file_reader = csv.reader(train_label_file)
        row = next(train_label_file_reader)
        for row in train_label_file_reader:
            if not use_all_data:
                if row[-1] != 'TEST':
                    if int(row[-1]) in training_folds:
                        train_labels[row[2]] = [float(row[3])]
                        train_ids.append(row[2])
                        train_img_txt_ids[row[2]] = row[1]
                    if int(row[-1]) in validation_folds:
                        val_labels[row[2]] = [float(row[3])]
                        val_ids.append(row[2])
                        val_img_txt_ids[row[2]] = row[1]
                if row[-1] == 'TEST':
                    test_labels[row[2]] = [float(row[3])]
                    test_ids.append(row[2])
                    test_img_txt_ids[row[2]] = row[1]
            else:
                train_labels[row[2]] = [float(row[3])]
                train_ids.append(row[2])
                train_img_txt_ids[row[2]] = row[1]                      
                val_labels[row[2]] = [float(row[3])]
                val_ids.append(row[2])
                val_img_txt_ids[row[2]] = row[1]

    print("Training and validation folds: ", training_folds, validation_folds)
    print("Total number of training labels: ", len(train_labels))
    print("Total number of training DICOM IDs: ", len(train_img_txt_ids))
    print("Total number of validation labels: ", len(val_labels))
    print("Total number of validation DICOM IDs: ", len(val_img_txt_ids))
    print("Total number of test labels: ", len(test_labels))
    print("Total number of test DICOM IDs: ", len(test_img_txt_ids))

    if use_all_data:
        return train_labels, train_img_txt_ids, val_labels, val_img_txt_ids

    # When use_test_data is True, test set will be returned for evaluation;
    # Otherwise, validation set will be returned (likely for cross validation)
    if use_test_data:
        return train_labels, train_img_txt_ids, test_labels, test_img_txt_ids
    else:
        return train_labels, train_img_txt_ids, val_labels, val_img_txt_ids


class RandomTranslateCrop(object):
    """Translate and crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size, shift_mean=0,
                 shift_std=200, rotation_mean=0, rotation_std=20):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.rotation_mean = rotation_mean
        self.rotation_std = rotation_std

    def __call__(self, image):
        image = self.__translate_2Dimage(image)
        #image = self.__rotate_2Dimage(image)
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size

        if new_h>h or new_w>w:
            raise ValueError('This image needs to be padded!')

        top = floor((h - new_h) / 2)
        down = top + new_h
        left = floor((w - new_w) / 2)
        right = left + new_w
        
        return image[top:down, left:right]

    def __translate_2Dimage(self, image):
        'Translate 2D images as data augmentation'
        h, w = image.shape[0:2]
        h_output, w_output = self.output_size[0:2]

        # Generate random Gaussian numbers for image shift as data augmentation
        shift_h = int(np.random.normal(self.shift_mean, self.shift_std))
        shift_w = int(np.random.normal(self.shift_mean, self.shift_std))
        if abs(shift_h) > 2 * self.shift_std:
            shift_h = 0
        if abs(shift_w) > 2 * self.shift_std:
            shift_w = 0

        # Pad the 2D image
        pad_h_length = max(0, float(h_output - h))
        pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_1 = pad_h_length_1 + max(shift_h , 0)
        pad_h_length_2 = pad_h_length_2 + max(-shift_h , 0)

        pad_w_length = max(0, float(w_output - w))
        pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_1 = pad_w_length_1 + max(shift_w , 0)
        pad_w_length_2 = pad_w_length_2 + max(-shift_w , 0)

        image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
                       'constant', constant_values=((0, 0), (0, 0)))

        return image

    def __rotate_2Dimage(self, image):
        'Rotate 2D images as data augmentation'

        # Generate a random Gaussian number for image rotation angle as data augmentation
        angle = np.random.normal(self.rotation_mean, self.rotation_std)
        if abs(angle) > 2 * self.rotation_std:
            angle = 0
            return image

        return ndimage.rotate(image, angle)


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        image = self.__pad_2Dimage(image)
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size

        if new_h>h or new_w>w:
            raise ValueError('This image needs to be padded!')

        top = floor((h - new_h) / 2)
        down = top + new_h
        left = floor((w - new_w) / 2)
        right = left + new_w
        
        return image[top:down, left:right]
    
    def __pad_2Dimage(self, image):
        'Pad 2D images to match output_size'
        h, w = image.shape[0:2]
        h_output, w_output = self.output_size[0:2]

        pad_h_length = max(0, float(h_output - h))
        pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding

        pad_w_length = max(0, float(w_output - w))
        pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding

        image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
                       'constant', constant_values=((0, 0), (0, 0)))

        return image

# Convert edema severity to one-hot encoding
def convert_to_onehot(severity):
    if int(severity) == 0:
        return [1,0,0,0]
    elif int(severity) == 1:
        return [0,1,0,0]
    elif int(severity) == 2:
        return [0,0,1,0]
    elif int(severity) == 3:
        return [0,0,0,1]
    elif int(severity) == -1:
        return [-1,-1,-1,-1]
    else:
        raise Exception("No other possibilities of one-hot labels are possible")

# Convert edema severity to ordinal encoding
def convert_to_ordinal(severity):
    if int(severity) == 0:
        return [0,0,0]
    elif int(severity) == 1:
        return [1,0,0]
    elif int(severity) == 2:
        return [1,1,0]
    elif int(severity) == 3:
        return [1,1,1]
    elif int(severity) == -1:
        return [-1,-1,-1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")


class CXRImageTextDataset(Dataset):
    
    def __init__(self, img_localdisk_data_dir, model_id , all_txt_tokens, 
                 all_txt_masks, all_txt_segments, all_txt_labels,
                 all_img_txt_ids, img_dir, all_img_labels, transform=None, 
                 output_channel_encoding='multilabel'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on an image sample.
        """
        self.all_txt_tokens = all_txt_tokens
        self.all_txt_masks = all_txt_masks
        self.all_txt_segments = all_txt_segments
        self.all_txt_labels = all_txt_labels
        self.all_img_txt_ids = all_img_txt_ids
        self.all_img_labels = all_img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.output_channel_encoding = output_channel_encoding
        self.img_format = '.png'

        print('Image directory: ', self.img_dir)

    def __len__(self):
        return len(self.all_img_txt_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = list(self.all_img_txt_ids.keys())[idx]


        img_path = os.path.join(self.img_dir, img_id+self.img_format)
        image = load_image(img_path)
        if self.transform:
            image = self.transform(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        img_label = self.all_img_labels[img_id]
        img_label = torch.tensor(img_label, dtype=torch.float32)
        
        txt_id = self.all_img_txt_ids[img_id]
        
        txt_tokens = torch.tensor(self.all_txt_tokens[txt_id], dtype=torch.long)
        txt_mask = torch.tensor(self.all_txt_masks[txt_id], dtype=torch.long)
        txt_segments = torch.tensor(self.all_txt_segments[txt_id], dtype=torch.long)
        if self.output_channel_encoding == 'multilabel':
            txt_label = torch.tensor(self.all_txt_labels[txt_id], dtype=torch.long)
        elif self.output_channel_encoding == 'multiclass':
            txt_label = torch.tensor(convert_to_onehot(self.all_txt_labels[txt_id]), dtype=torch.long)
        # txt_label is ordinal in the case of multilabel, otherwise one-hot.
        # img_label ranges from 0 to 3 (not encoded yet)

        report_id = int(txt_id)

        sample = [image, img_label, txt_tokens, txt_mask, txt_segments, txt_label, report_id]

        return sample


# Load an .npy or .png image 
def load_image(img_path):
    if img_path[-3:] == 'npy':
        image = np.load(img_path)
    if img_path[-3:] == 'png':
        image = io.imread(img_path)
        image = image.astype(np.float32)
        image = image/np.max(image)
    return image


