import matplotlib.pyplot as plt
import pickle
import numpy as np
import os.path
import glob

def unpickle(path):
    with open(path, "rb") as f:
        dict_data = pickle.load(f, encoding="bytes")
    return dict_data 

def count(path):
    data = {}
    paths = glob.glob(path + "/*")
    for folder in paths:
        class_name = os.path.basename(folder)
        amount = len(glob.glob(folder + "/*.png"))
        data.update({class_name: amount})
    return data

savepath = "plot_image/data_bar.jpg"
DATAROOT = "app/data"

train = count(DATAROOT + "/train")
test = count(DATAROOT + "/test")

name = train.keys()
train_amount = [train[s] for s in name]
test_amount = [test[s] for s in name]

#プロット
w=0.4
x=np.arange(len(name))
plt.title("amount", fontsize=15)
plt.xlabel("class")
plt.ylabel("images")
plt.tick_params(labelsize=6)
plt.bar(x, train_amount, color="b", width=0.4, label="train")
plt.bar(x + w, test_amount, color="r", width=0.4, label="test")
plt.xticks(x + w/2, name)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.subplots_adjust(right=0.8)

plt.savefig(savepath)