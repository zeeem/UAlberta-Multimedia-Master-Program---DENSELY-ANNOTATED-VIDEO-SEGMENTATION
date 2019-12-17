"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
from PIL import Image
import os
import numpy as np
import sys
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import imgaug.augmenters as iaa

class Dataset:
    def __init__(self, train_list, test_list, database_root, store_memory=True, data_aug=False, new_aug=False):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        database_root: Path to the root of the Database
        store_memory: True stores all the training images, False loads at runtime the images
        Returns:
        """
        if not store_memory and data_aug:
            sys.stderr.write('Online data augmentation not supported when the data is not stored in memory!')
            sys.exit()
        # Define types of data augmentation
        data_aug_scales = [0.5, 0.8, 1]
        data_aug_flip = True
        
        #checking aug bool value
        print('new augmentation value is : ')
        print(new_aug)
        
        
        if(new_aug):
            #extra augmentation
            data_aug_brightness = True
            data_aug_addnoise = False
        else:
            data_aug_brightness = False
            data_aug_addnoise = False

        # Load training images (path) and labels
        print('Started loading files...')
        if not isinstance(train_list, list) and train_list is not None:
            with open(train_list) as t:
                train_paths = t.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []
        if not isinstance(test_list, list) and test_list is not None:
            with open(test_list) as t:
                test_paths = t.readlines()
        elif isinstance(test_list, list):
            test_paths = test_list
        else:
            test_paths = []
        self.images_train = []
        self.images_train_path = []
        self.labels_train = []
        self.labels_train_path = []
        for idx, line in enumerate(train_paths):
            if store_memory:
                img = Image.open(os.path.join(database_root, str(line.split()[0])))
                img.load()
                label = Image.open(os.path.join(database_root, str(line.split()[1])))
                label.load()
                label = label.split()[0]
                j = 0   
                if data_aug:
                    if idx == 0: sys.stdout.write('Performing the data augmentation')
                    for scale in data_aug_scales:
                        img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
                        img_sc = img.resize(img_size)
                        label_sc = label.resize(img_size)
                        self.images_train.append(np.array(img_sc, dtype=np.uint8))
                        self.labels_train.append(np.array(label_sc, dtype=np.uint8))
                        if(data_aug_addnoise):
                            im_lap_noise = img_to_array(img_sc)
                            seq = iaa.Sequential([iaa.AdditiveLaplaceNoise(scale=0.9*255)]) #scale was 11 before
                            image_noise_np = seq.augment_image(im_lap_noise).astype('uint8')
                            # im_lap_noise_sq = Image.fromarray(seq.augment_image(im_lap_noise))
                            # im_lap_noise_sq.save('./aug_image/test_save/aug-im-noise' + str(j), "JPEG")
                            self.images_train.append(np.array(image_noise_np, dtype=np.uint8))
                            self.labels_train.append(np.array(label_sc, dtype=np.uint8))
                           
                        if data_aug_flip:
                            img_sc_fl = img_sc.transpose(Image.FLIP_LEFT_RIGHT)
                            label_sc_fl = label_sc.transpose(Image.FLIP_LEFT_RIGHT)
                            self.images_train.append(np.array(img_sc_fl, dtype=np.uint8))
                            self.labels_train.append(np.array(label_sc_fl, dtype=np.uint8))
                        if data_aug_brightness:   
                            data_img = img_to_array(img_sc)
                            br_samples = np.expand_dims(data_img, 0)
                            datagen = ImageDataGenerator(brightness_range=[0.2,1.7])
                            it = datagen.flow(br_samples, batch_size=1)
                            for i in range(9):
                                batch = it.next()
                                image_np = batch[0].astype('uint8')
                                aug_im = Image.fromarray(image_np)
                                aug_im.save('./aug_image/test_save/aug-im-' + str(j), "JPEG")
                                self.images_train.append(np.array(image_np, dtype=np.uint8))
                                self.labels_train.append(np.array(label_sc, dtype=np.uint8))
                                j = j+1
                    
                else:
                    if idx == 0: sys.stdout.write('Loading the data')
                    self.images_train.append(np.array(img, dtype=np.uint8))
                    self.labels_train.append(np.array(label, dtype=np.uint8))
                if (idx + 1) % 50 == 0:
                    sys.stdout.write('.')
            self.images_train_path.append(os.path.join(database_root, str(line.split()[0])))
            self.labels_train_path.append(os.path.join(database_root, str(line.split()[1])))
        sys.stdout.write('\n')
        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)

        # Load testing images (path) and labels
        self.images_test = []
        self.images_test_path = []
        for idx, line in enumerate(test_paths):
            if store_memory:
                self.images_test.append(np.array(Image.open(os.path.join(database_root, str(line.split()[0]))),
                                                 dtype=np.uint8))
                if (idx + 1) % 1000 == 0:
                    print('Loaded ' + str(idx) + ' test images')
            self.images_test_path.append(os.path.join(database_root, str(line.split()[0])))
        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = max(len(self.images_train_path), len(self.images_train))
        self.test_size = len(self.images_test_path)
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)
        self.store_memory = store_memory

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        Returns in testing:
        images: None if store_memory=False, Numpy array of the image if store_memory=True
        path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                else:
                    images = [self.images_train_path[l] for l in idx]
                    labels = [self.labels_train_path[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path[l] for l in old_idx]
                    labels_1 = [self.labels_train_path[l] for l in old_idx]
                    images_2 = [self.images_train_path[l] for l in idx]
                    labels_2 = [self.labels_train_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                self.train_ptr = new_ptr
            return images, labels
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                if self.store_memory:
                    images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                paths = self.images_test_path[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if self.store_memory:
                    images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                paths = self.images_test_path[self.test_ptr:] + self.images_test_path[:new_ptr]
                self.test_ptr = new_ptr
            return images, paths
        else:
            return None, None

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width
