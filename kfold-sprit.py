import glob, os
from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools
import cv2
import numpy as np
import random, time
import matplotlib.pyplot as plt
import collections

def plot_data(train_y, val_y, savepath):
    name = sorted(list(set(train_y)))
    train_dict = collections.Counter(train_y)
    val_dict = collections.Counter(val_y)
    train_amount = [train_dict[s] for s in name]
    val_amount = [val_dict[s] for s in name]
    w=0.4
    x=np.arange(len(name))
    plt.title("amount", fontsize=15)
    plt.xlabel("class")
    plt.ylabel("images")
    plt.tick_params(labelsize=6)
    plt.bar(x, train_amount, color="b", width=0.4, label="train")
    plt.bar(x + w, val_amount, color="r", width=0.4, label="validation")
    plt.xticks(x + w/2, name)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.subplots_adjust(right=0.8)

    plt.savefig(savepath)
    plt.close()

def kfold_sprit(split_size, path, saveroot):
    data = {}
    root = os.path.join(path)
    labels = os.listdir(root)
    kf = KFold(n_splits=split_size)
    train_image = []
    train_label = []
    for label in labels:
        images = os.listdir(os.path.join(root, label))
        train_image.append(images)
        train_label.append([label for i in range(len(images))])
    train_image = list(itertools.chain.from_iterable(train_image))
    train_label = list(itertools.chain.from_iterable(train_label))
    seed = int(time.time())
    random.seed(seed)
    random.shuffle(train_image)
    random.seed(seed)
    random.shuffle(train_label)

    for i, idx in enumerate(kf.split(train_image, train_label)):
        train_idx, val_idx = idx
        train_x = np.array(train_image)[train_idx]
        train_y = np.array(train_label)[train_idx]
        val_x = np.array(train_image)[val_idx]
        val_y = np.array(train_label)[val_idx]
        savepath = saveroot + "/round{}.png".format(i)
        plot_data(train_y=train_y, val_y=val_y, savepath=savepath)




DATAROOT = "app/data"
SAVEROOT = "plot_image"
split_size = 10

kfold_sprit(split_size=split_size, path=DATAROOT+"/train", saveroot=SAVEROOT)