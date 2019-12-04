import pickle
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

class Loader:

    def __init__(self, root):
        self.dataRoot = root

    def main(self, name):
        a ='test' if name == 'test_batch' else 'train'
        self.saveRoot = self.dataRoot.replace(os.path.basename(self.dataRoot), a)
        # ラベルデータの読み込み
        labels = self.unpickle(path=self.dataRoot + '/batches.meta')
        # イメージデータの読み込み
        print('【展開】', name, 'ディレクトリ')
        images = self.unpickle(path=self.dataRoot + '/'+ name)
        # ファイルの展開
        self.extractImage(labels=labels, images=images)

    def extractImage(self, labels, images):
        # 属性抽出
        label_names = labels[b'label_names']
        filenames = images[b'filenames']
        image_labels = images[b'labels']
        image_data = images[b'data']
        # ファイル格納先のフォルダ作成
        for i in label_names:
            dir = os.path.join(self.saveRoot, i.decode('utf-8'))
            os.makedirs(dir,exist_ok=True)
        # 進捗バーの設定
        pbar = tqdm(range(len(filenames)))
        # イメージファイルの展開
        for i in pbar:
            # 進捗バーの説明
            pbar.set_description("Unzip %s" % i)
            # イメージファイルの属性取得
            filename = filenames[i].decode('utf-8')
            image_label = label_names[image_labels[i]].decode('utf-8')
            image = image_data[i]
            # 保存パス
            file_path = os.path.join(self.saveRoot, image_label, filename)
            # イメージファイルの保存
            reshaped_array = np.reshape(image, [3, 32, 32]).transpose(1, 2, 0)
            im = Image.fromarray(reshaped_array)
            self.saveImage(path=file_path, data=im)

    def saveImage(self, path, data):
        with open(path, mode='wb') as f:
            data.save(f)
    
    def unpickle(self, path):
        with open(path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        return dict