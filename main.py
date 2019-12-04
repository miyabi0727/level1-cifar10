from app.src.cifar10Loader import Loader
import numpy as np

def main():
    # イメージファイル展開関数
    loader = Loader(root='./app/data/cifar-10-batches-py')
    # 訓練データのファイル展開
    for i in np.arange(1,6):
        loader.main(name='data_batch_' + str(i))
    # テストデータのファイル展開
    loader.main(name='test_batch')

if __name__ == "__main__":
    main()