import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

def load_image(path):
    """Read an image as a numpy array"""
    return np.array(Image.open(path))

def resolve_single(model, lr):
    """Pass a single image to the model"""
    return resolve(model, np.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    """Pass a batch of images to the model"""
    lr_batch = lr_batch.astype("float32")
    sr_batch = model(lr_batch)
    sr_batch = np.clip(sr_batch, 0, 255)
    sr_batch = np.around(sr_batch)
    sr_batch = sr_batch.astype("uint8")
    return sr_batch

def bicubic_upsample(lr):
    """Perform Bicubic interpolation on an image"""
    h, w, _ = lr.shape
    return cv2.resize(lr, (w*4, h*4), interpolation=3)

#Evaluation Metrics
def sklearn_cosine(img1, img2):
    """Compute Cosine Similarity from scikitlearn"""
    im1=cv2.cvtColor(img1  , cv2.COLOR_RGB2YUV) 
    im2=cv2.cvtColor(img2  , cv2.COLOR_RGB2YUV) 
    return cosine_similarity([(im1[:,:,0]).flatten()], [(im2[:,:,0]).flatten()])[0][0]

def scipy_cosine(img1, img2):
    """Compute Cosine Similarity from scipy"""
    im1=cv2.cvtColor(img1  , cv2.COLOR_RGB2YUV) 
    im2=cv2.cvtColor(img2  , cv2.COLOR_RGB2YUV) 
    return 1. - cdist([(im1[:,:,0]).flatten()],[ (im2[:,:,0]).flatten()], 'cosine')[0][0]

def SSIM_y_channel(img1, img2):
    """Calculae SSIM on the Y channel of YUV Image"""
    im1=cv2.cvtColor(img1  , cv2.COLOR_RGB2YUV) 
    im2=cv2.cvtColor(img2  , cv2.COLOR_RGB2YUV) 
    sim = structural_similarity(im1[:,:,0],im2[:,:,0])
    return sim


def PSNR_y_channel(true_img, fake_img):
    """Calculae PSNR on the Y channel of YUV Image"""
    im_true=cv2.cvtColor(true_img  , cv2.COLOR_RGB2YUV) 
    im_fake=cv2.cvtColor(fake_img  , cv2.COLOR_RGB2YUV) 
    psnr= peak_signal_noise_ratio(im_true[:,:,0],im_fake[:,:,0])
    return psnr
