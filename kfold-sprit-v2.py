import os, csv
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot(hist_idx, label, n_splits, length, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    cmap_cv = plt.cm.coolwarm
    cmap_data = plt.cm.Paired
    # Generate the training/testing visualizations for each CV split
    for ii, idx in enumerate(hist_idx):
        tr, tt = idx
        indices = np.array([np.nan] * length)
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
    # Plot the data classes and groups at the end
    ax.scatter(range(length), [ii + 1.5] * length,
               c=label, marker='_', lw=lw, cmap=cmap_data)
    # Formatting
    y_labels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=y_labels,
           xlabel='index', ylabel="iteration",
           ylim=[n_splits+1.2, -.2], xlim=[-1000, 50000])
    ax.set_title('StratifiedKFold', fontsize=15)
    return ax

def label_creat():
    with open('label.csv', newline='') as f:
        data = csv.reader(f)
        label = [x for x in data]
    return dict(label)

def open_image(path):
    img = Image.open(path)
    image = np.asarray(img).astype("f2")/255
    img.close()
    return image


def kfold_sprit(kfold_size, path, saveroot):
    savepath = saveroot + "/kfold.png"
    root = os.path.join(path)
    label_dict = label_creat()
    train_image = []
    train_label = []
    for label in list(label_dict.keys()):
        images = os.listdir(os.path.join(root, label))
        for path in tqdm(images):
            image = open_image(os.path.join(root, label, path))
            train_image.append(image)
            train_label.append(int(label_dict[label]))
            del image
    #メモリ削減
    image_np = np.array(train_image)
    label_np = np.array(train_label)
    del train_image
    del train_label
    skf = StratifiedKFold(n_splits=kfold_size)
    hist_idx = []
    for idx in skf.split(image_np, label_np):
        train_idx, val_idx = idx
        hist_idx.append(idx)
        train_x = image_np[(train_idx)]
        train_y = label_np[(train_idx)]
        val_x = image_np[(val_idx)]
        val_y = label_np[(val_idx)]
        del train_idx, val_idx, idx
    fig, ax = plt.subplots()
    ax = plot(hist_idx=hist_idx, label=label_np, n_splits=kfold_size, length=len(image_np))
    plt.savefig(savepath)


DATAROOT = "app/data"
SAVEROOT = "plot_image"
kfold_size = 10
kfold_sprit(kfold_size=kfold_size, path=DATAROOT+"/train", saveroot=SAVEROOT)