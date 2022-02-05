import cv2
import numpy as np
from glob import glob
from tensorflow.keras.utils import Sequence


class Div2kLoader(Sequence):
    """Creates A generator object for DIV2K dataset that can be used to train keras models.
    
    Attributes
    ----------
    path: str
        Path to the location of the training data.
    
    patch_size: int
        Number of images per patch.
        Default=32
        
    shuffle: bool
        Whether to shuffle the dataset after each epoch.
        Default=True

    load_all_data: bool
        Whether to load the whole dataset in memory.
        Default=False

    Methods
    -------

    load_batch()
        Loads a single batch of data. Mainly used for SRGAN training
    """

    def __init__(self, path: str, batch_size: int = 32, shuffle: bool = True, load_all_data: bool = False):
      self.batch_size = batch_size   
      self.shuffle = shuffle
      self.load_all_data = load_all_data

      #Get all paths of training images:
      self.train_hr_paths = sorted(glob(f"{path}/HR/*"))
      self.train_lr_paths = sorted(glob(f"{path}/LR/*"))
      self.indexes = np.arange(len(self.train_hr_paths))

      self.batch_no = 0 #For the load_patch method.

      if self.load_all_data:
        self.hr_images = np.array([cv2.imread(path) for path in self.train_hr_paths])
        self.lr_images = np.array([cv2.imread(path) for path in self.train_lr_paths])


    def __len__(self):
      """Denotes the number of batches per epoch"""
      return int(np.floor(len(self.indexes)/self.batch_size))
    
    def on_epoch_end(self):
      """Determine what happens after each epoch"""
      #Shuffle indexes after each epoch
      self.indexes = np.arange(len(self.train_hr_paths))
      if self.shuffle:
          np.random.shuffle(self.indexes)

    def __getitem__(self, index):
      """Generates one patch of data"""
      indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      if self.load_all_data:
        hr_images = self.hr_images[indices]
        lr_images = self.lr_images[indices]
      else:         
        hr_images = np.array([cv2.imread(self.train_hr_paths[i]) for i in indices])
        lr_images = np.array([cv2.imread(self.train_lr_paths[i]) for i in indices])

      return lr_images, hr_images
  
    def load_batch(self):
        """Used for training SRGAN"""
        if 2*self.batch_no > self.__len__():
            self.on_epoch_end()
            self.batch_no = 0
            
        lr_images, hr_images = self.__getitem__(self.batch_no)
        self.batch_no += 1
        
        return lr_images, hr_images