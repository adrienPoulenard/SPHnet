from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os


def save_model(path, model):
    # serialize model to JSON
    model_json = model.to_json()
    model_path = 'model.json'
    model_path = os.path.join(path, model_path)
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    weights_path = 'weights.h5'
    weights_path = os.path.join(path, weights_path)
    model.save_weights(weights_path)
    print("Saved model to disk")


def save_training_acc(path, hist):
    train_acc = hist.history['acc']
    with open(os.path.join(path, 'train_acc.txt'), 'w') as f:
        for item in train_acc:
            f.write("%s\n" % item)
    val_acc = hist.history['val_acc']
    with open(os.path.join(path, 'val_acc.txt'), 'w') as f:
        for item in val_acc:
            f.write("%s\n" % item)


def save_list(path, list):
    with open(path, 'w') as f:
        for item in list:
            f.write("%s\n" % item)
        f.close()


def save_tensor(path, x):
    np.savetxt(path, x.flatten(), delimiter=' ', fmt='%f')


def save_matrix(path, x, dtype=np.float32):
    if dtype is np.float32:
        np.savetxt(path, x, delimiter=' ', fmt='%f')
    if dtype is np.int32:
        np.savetxt(path, x.astype(np.int32), delimiter=' ', fmt='%i')


def create_dir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)





