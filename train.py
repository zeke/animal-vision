from fastai.vision.all import *
from pathlib import Path
from shared import label_func

# Based on https://docs.fast.ai/tutorial.vision.html

print("Downloading data set...")
path = untar_data(URLs.PETS)

# Limit number of images to make it train quickly on CPU
files = get_image_files(path/"images")[:500]

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
# Save exported files in current directory
learn.path = Path.cwd()

learn.fine_tune(1)

# Creates export.pkl
learn.export()