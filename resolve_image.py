from utils import load_image, resolve_single
from matplotlib.pyplot import imsave
from models import edsr
import ntpath
import sys

def resolve_and_save(path, model):
    lr = load_image(path)
    sr = resolve_single(model, lr)
    imsave(f"results/{ntpath.basename(path)}",sr)

def main():
    if len(sys.argv) == 1:
        raise RuntimeError("An image path must be passed in order to process it!")
    else:
        path = sys.argv[1]

    model = edsr()
    model.load_weights("model_weights/EDSR_X4_SRGAN-final.h5")
    resolve_and_save(path, model)

if __name__ == '__main__':
    main()