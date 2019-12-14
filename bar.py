import matplotlib.pyplot as plt
import pickle
import numpy as np

def unpickle(path):
    with open(path, "rb") as f:
        dict_data = pickle.load(f, encoding="bytes")
    return dict_data 

DATAROOT = "../DETASET/cifar-10-batches-py"
train_labels = np.ndarray([])
test_labels = np.ndarray([])
#trainイメージラベル取得
for i in range(1,6):
    train_dict = unpickle(DATAROOT + "/data_batch_" + str(i))
    train_labels = np.insert(train_labels, train_labels.size, train_dict[b"labels"]).astype("int")

#testイメージラベル取得
test_dict = unpickle(DATAROOT + "/test_batch")[b"labels"]
test_labels = np.insert(test_labels, test_labels.size, test_dict).astype("int")

#classごとのカウント
train_amont = np.bincount(train_labels)
test_amont = np.bincount(test_labels)

#ラベル名取得
meta = unpickle(DATAROOT + "/batches.meta")[b"label_names"]
label_name = [s.decode("utf8") for s in meta]

#プロット
plt.subplot(1,2,1)
plt.title("train", fontsize=15)
plt.xlabel("class")
plt.ylabel("images")
plt.tick_params(labelsize=7)
plt.bar(label_name, train_amont)

plt.subplot(1,2,2)
plt.title("test", fontsize=15)
plt.xlabel("class")
plt.ylabel("images")
plt.tick_params(labelsize=7)
plt.ylim(ymax=5000, ymin=0)
plt.bar(label_name, test_amont)

plt.show()