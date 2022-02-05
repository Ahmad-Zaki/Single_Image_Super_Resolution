import cv2
import pickle
from glob import glob
import os
from tensorflow.keras.utils import get_file

class Div2k:
    """
    Class that can download the specified version of DIV2k dataset and transform it to a pickle file if required.

    Parameters
    ----------
    path : str
      The path to the folder containing the dataset on your disk (Required if you want to pickle the data)

    subsets : list, tuple
      download training or validation or both.
      Default=["train"]

    downgrade : str
      the type of downgrade on the low resolution images, can be either "bicubic" or "unknown"
      Defalut="bicubic"

    scale: int
      the downscale of the low resolution images.
      Default=4

    Methods
    -------
    download()
      Fetch and download the desired split of DIV2K dataset.

    pickle()
      Read the dataset and sava each subset in pickle file(s).
    """

    def __init__(self, path: str, subsets: list = None, downgrade: str = "bicubic", scale: int = 4):
        self._DL_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
        self.path = path
        self.subsets = subsets if subsets else ["train"]
        self.downgrade = downgrade
        self.scale = scale

        self.data_dirs = {f"{subset}_hr": f"{self.path}/div2k/DIV2K_{subset}_HR" for subset in self.subsets}
        self.data_dirs.update({f"{subset}_lr": f"{self.path}/div2k/DIV2K_{subset}_LR_{downgrade}/X{4}" for subset in self.subsets})
    
    
    def download(self):
        """
        Fetch and download the desired split of DIV2K dataset.
        """

        for split in self.subsets:
            HR_DATA = f"DIV2K_{split}_HR"
            LR_DATA = f"DIV2K_{split}_LR_{self.downgrade}_X{self.scale}"
    
            for data in [HR_DATA, LR_DATA]:
                path_to_downloaded_file = get_file(fname = f"{data}.zip",
                                           cache_dir = self.path,
                                           cache_subdir = self.path + "/div2k",
                                           origin = f"{self._DL_URL}{data}.zip",
                                           extract = True,
                                           archive_format = 'zip')
        
                #Remove the zip file after extraction to save disk space:
                os.remove(path_to_downloaded_file)
      
    def pickle(self):
        """
        Read the dataset and sava each subset in pickle file(s).
        """
        
        for subset_name, subset_path in self.data_dirs.items():
            i = 0 #Images counter
            images = list()
            for image_path in sorted(glob(f"{subset_path}/*.png")):
                i += 1
                #Read the image:
                images.append(cv2.imread(image_path)) 
                #Remove the image from disk to save space:
                os.remove(image_path)
                
                #Save every 100 images in a pickle file:
                if i % 100 == 0:
                    with open(f"{subset_path}/{subset_name}_{i//100}.p","wb") as pickle_file:
                        pickle.dump(images,pickle_file)
                    images = list()

if __name__ == '__main__':
    PATH = r"./datasets"
    div2k = Div2k(PATH, ["train", "valid"])
    div2k.download()