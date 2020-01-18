import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import os, csv
from PIL import Image
import numpy as np
import pickle

def unpickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def one_hot(label):
    oh = np_utils.to_categorical(label)
    return oh

def label_creat():
    with open('label.csv', newline='') as f:
        data = csv.reader(f)
        label = [x for x in data]
    return dict(label)

def open_image(path):
    img = Image.open(path)
    image = np.asarray(img).astype("f4")/255
    img.close()
    return image

def Classification(DATAROOT):
    test_root = os.path.join(DATAROOT, "test")
    label_dict = label_creat()
    test_image = []
    test_label = []
    for label in list(label_dict.keys()):
        test = os.listdir(os.path.join(test_root, label))
        for path in test:
            image = open_image(os.path.join(test_root, label, path))
            test_image.append(image)
            test_label.append(int(label_dict[label]))
    #メモリ削減
    del image
    test_image = np.array(test_image)
    #one-hot-labelに
    test_label = np.array(one_hot(label=test_label))
    test = test_image, test_label

    corss_val_root = os.path.join(DATAROOT, "cross-validation")
    rounds = os.listdir(corss_val_root)
    for round_name in rounds:
        path_round = os.path.join(corss_val_root, round_name)
        print(os.path.join(path_round, "train_x.pickle"))
        train_x = unpickle(os.path.join(path_round, "train_x.pickle"))
        train_y = unpickle(os.path.join(path_round, "train_y.pickle"))
        val_x = unpickle(os.path.join(path_round, "val_x.pickle"))
        val_y = unpickle(os.path.join(path_round, "val_y.pickle"))
        train = train_x, train_y
        val = val_x, val_y
        run(train=train, val=val, test=test, round_name=round_name)

def run(train, val, test, round_name="round0"):
    x_train, y_train = train
    val_x, val_y = val
    test_x, test_y = test
    batch_size = 64
    num_classes = 10
    epochs = 10
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_x, val_y),
                shuffle=True)
    scores = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # Save model and weights
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model_{}.h5'.format(round_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
DATAROOT = "app/data"
Classification(DATAROOT=DATAROOT)