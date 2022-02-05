import cv2
from data_downloader import Div2k
from glob import glob
from models import edsr, srgan_discriminator
from model_trainer import SrganTrainer
from pathlib import Path
from preprocessing import data_preprocessing
import sys

def main():
    if len(sys.argv) == 1:
        # Download DIV2K dataset:
        PATH = r"./datasets"
        div2k = Div2k(PATH, ["train", "valid"])
        div2k.download()

        # Preprocess the dataset:
        hr_images = sorted(glob(r"datasets/div2k/DIV2K_train_HR/*.png"))
        lr_images = sorted(glob(r"datasets/div2k/DIV2K_train_LR_bicubic/X4/*.png"))

        Path('datasets/preprocessed_data/HR').mkdir(parents=True)
        Path('datasets/preprocessed_data/LR').mkdir(parents=True)
        write_hr = r"datasets/preprocessed_data/HR/"
        write_lr = r"datasets/preprocessed_data/LR/"

        degree = 0
        for i in range(len(hr_images)):
            print(f"image #{i}:", end="")
            if i%100 == 0: degree += 10
            hr_img = cv2.imread(hr_images[i])
            lr_img = cv2.imread(lr_images[i])
            d = data_preprocessing(hr_img, lr_img,
                                write_hr, write_lr,
                                i, i,
                                (256, 256), (64, 64),
                                degree=degree, threshold=0.6)
            d.generate_images()
            print("done")

        data_path = r"datasets/preprocessed_data/"
    else:
        data_path = sys.argv[1]

    # Start Training:
    generator = edsr()
    discriminator = srgan_discriminator()
    gan = SrganTrainer(generator,
                       discriminator,
                       data_path=data_path,
                       load_all_data=False)

    weights_path = gan.trainGenerator(epochs=150,
                                      batch_size=32)

    gan.train_gan(weights_path,
                  steps=2e5,
                  batch_size=16)

if __name__=="__main__":
    main()