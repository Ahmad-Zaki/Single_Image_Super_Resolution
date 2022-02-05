from scipy import stats
from scipy import ndimage
from glob import glob
import cv2
import matplotlib.pyplot as plt

class data_preprocessing:
    def __init__(self, hr_img, lr_img, write_hr, write_lr, hr_prefix, lr_prefix, hr_shape, lr_shape, degree=10, threshold=0.8):
      self.hr_img = hr_img
      self.lr_img = lr_img
      self.write_hr = write_hr
      self.write_lr = write_lr
      self.hr_prefix = hr_prefix
      self.lr_prefix = lr_prefix
      self.hr_shape = hr_shape
      self.lr_shape = lr_shape 
      self.degree = degree
      self.threshold = threshold
    
    def generate_images(self):
      '''
      Save Genrated images to predefined path (write_path)
      '''
      good_hr_imgs, good_lr_imgs = data_preprocessing.create_images(self.hr_img, self.lr_img, self.hr_shape, self.lr_shape, self.degree, self.threshold)
      for i in range(len(good_hr_imgs)):
        hr_name = f"{self.hr_prefix}_{i}.png"
        lr_name = f"{self.lr_prefix}_{i}.png"
        
        cv2.imwrite(self.write_hr+hr_name, good_hr_imgs[i])
        cv2.imwrite(self.write_lr+lr_name, good_lr_imgs[i])
        
    @staticmethod
    def create_images(hr_img, lr_img, hr_shape, lr_shape, degree, threshold):
      hr_imgs = []
      lr_imgs = []
      good_hr_imgs = []
      good_lr_imgs = []

      # Append roteted image 
      hr_imgs.append(data_preprocessing.center_after_rotation(hr_img, hr_shape[0], hr_shape[1], degree))
      lr_imgs.append(data_preprocessing.center_after_rotation(lr_img, lr_shape[0], lr_shape[1], degree))

      # Append Cropped images 
      hr_imgs += data_preprocessing.divide_img(hr_img, hr_shape[0], hr_shape[1])
      lr_imgs += data_preprocessing.divide_img(lr_img, lr_shape[0], lr_shape[1])

      # Loop over imgs to filter images
      for i in range(len(hr_imgs)):
        # check if image contains one color or 10% of the image one color, if False so do normalization or standarization or nothing 
        new_img = data_preprocessing.filter_image(hr_imgs[i], threshold)
        if type(new_img) != int:
          good_hr_imgs.append(new_img)
          good_lr_imgs.append(lr_imgs[i])
                
      return good_hr_imgs, good_lr_imgs
    
    @staticmethod
    def filter_image(img, threshold=0.8):
      '''
      Return 0 if image will removed 
      Return image if image is balanced
      If image is unblanced apply Histogram Equalization on the unbalanced channel, then return the new image
      '''
      for i in range(3):
          pixel_values = img[:, :, i].flatten()
          mode_count = stats.mode(pixel_values)[1]
          color_proportion = mode_count / float(pixel_values.size)
          if color_proportion >= threshold:
            plt.imshow(img)
            plt.show();
            return 0
      return img
        
    @staticmethod
    def divide_img(img, height, width):
      '''
      Divide the picture (img) into (height x width) different images 
      '''
      images = []
      img_height = int(img.shape[0] / height)
      img_width = int(img.shape[1] / width)
      for i in range(img_height):
          for j in range(img_width):
              h = i*height
              w = j*width
              new_img = img[h:height+h, w:width+w, :]
              if new_img.size == (height * width * 3):
                  images.append(new_img)
      return images

    @staticmethod
    def center_after_rotation(img, h, w, degree=10):
      '''
      Apply rotation on the image then, crop the center of the image
      degree: Rotation degree  , expected value range is around [-25, 25]
      '''
      # Rotate image 
      rotated = ndimage.rotate(img, degree)
      
      # Get Center of the image 
      center = img.shape
      x = (center[1] / 2) - (w / 2)
      y = (center[0] / 2) - (h / 2)

      # Crop image and return
      return rotated[int(y):int(y+h), int(x):int(x+w)]
  
if __name__ == "__main__":
    hr_images = sorted(glob(r"datasets/div2k/DIV2K_train_HR/*.png"))
    lr_images = sorted(glob(r"datasets/div2k/DIV2K_train_LR_bicubic/X4/*.png"))
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