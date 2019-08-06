from data_providers.classifiaction_provider import ClassificationProvider as ClassificationProvider
import keras
import os
from time import time
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.save_model import save_matrix, create_dir, save_model
from methods.classification3_methods import methods_list
from data_providers.classification_datasets import datsets_list

datasets = datsets_list
methods = methods_list



MODELS_DIR = 'C:/Users/adrien/Documents/models'
RESULTS_DIR = 'C:/Users/adrien/Documents/results'

WEIGHTS_PATH = None
# WEIGHTS_PATH = 'C:/Users/adrien/Documents/models'
# WEIGHTS_PATH = os.path.join(WEIGHTS_PATH, 'modelnet40rotated_augmented/btree_inv_conv_2019_08_01_15_53_54/weights.h5')

assert(os.path.isdir(MODELS_DIR))
assert(os.path.isdir(RESULTS_DIR))
SAVE_MODELS = True

num_points = 1024
batch_size = 32
num_epochs = 250
SHUFFLE = True

def load_dataset(dataset):

    train_files_list = dataset['train_files_list']
    val_files_list = dataset['val_files_list']
    test_files_list = dataset['test_files_list']

    train_data_folder = dataset['train_data_folder']
    val_data_folder = dataset['val_data_folder']
    test_data_folder = dataset['test_data_folder']

    train_preprocessing = dataset['train_preprocessing']
    val_preprocessing = dataset['val_preprocessing']
    test_preprocessing = dataset['test_preprocessing']

    num_classes = dataset['num_classes']
    classes = dataset['classes']

    train_provider = ClassificationProvider(files_list=train_files_list,
                                            data_path=train_data_folder,
                                            n_classes=num_classes,
                                            n_points=num_points,
                                            batch_size=batch_size,
                                            preprocess=train_preprocessing,
                                            shuffle=SHUFFLE,
                                            classes=classes)

    val_provider = ClassificationProvider(files_list=val_files_list,
                                          data_path=val_data_folder,
                                          n_classes=num_classes,
                                          n_points=num_points,
                                          batch_size=batch_size,
                                          preprocess=val_preprocessing,
                                          shuffle=SHUFFLE,
                                          classes=classes)

    test_provider = ClassificationProvider(files_list=test_files_list,
                                           data_path=test_data_folder,
                                           n_classes=num_classes,
                                           n_points=num_points,
                                           batch_size=batch_size,
                                           preprocess=test_preprocessing,
                                           shuffle=False,
                                           classes=classes)

    return train_provider, val_provider, test_provider


def train(method, train_provider, val_provider):
    num_classes = train_provider.n_classes

    classifier = method['arch'](method=method)
    # loss = classifier.get_loss()
    bn_momentum = 0.5
    if 'bn_decay' in method['config']:
        bn_momentum = method['config']['bn_decay']
    classifier = classifier.get_network_model(num_classes=num_classes,
                                              num_points=num_points,
                                              batch_size=batch_size,
                                              bn_decay=bn_momentum)

    loss = keras.losses.categorical_crossentropy

    classifier.compile(loss=loss,
                       optimizer='adam',
                       metrics=['accuracy'])

    classifier.summary()

    if WEIGHTS_PATH is None:
        hist = classifier.fit_generator(generator=train_provider,
                                        validation_data=val_provider,
                                        epochs=num_epochs,
                                        callbacks=[],
                                        verbose=2)
    else:
        classifier.load_weights(WEIGHTS_PATH)
        hist = 0


    return classifier, hist

def test(classifier, test_provider, method=None, dataset=None):
    x, y = test_provider.get_data()
    y = np.reshape(y, newshape=(-1, ))
    y_pred = classifier.predict_generator(test_provider)
    # y_pred = classifier.predict(x, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=-1)
    y = y[:y_pred.shape[0], ...]
    acc = np.equal(y, y_pred)
    acc = acc.astype(np.float32)
    acc = np.mean(acc)
    if method is not None and dataset is not None:
        print(method['name'] + ' test_acc on ' + dataset['name'] + ' dataset is ', acc)
    conf_mat = confusion_matrix(y, y_pred)
    return conf_mat, acc

def save_model_(dir, method, model, timestamp):
    folder = os.path.join(dir, method['name'] + '_' + timestamp)
    save_model(model, folder)

def save_results(dir, hist, conf_mat, test_acc, train_time, test_time):
    folder = dir
    save_matrix(os.path.join(folder, 'confusion_matrix.txt'), conf_mat)
    # save_training_acc(folder, hist)
    save_matrix(os.path.join(folder, 'test_acc.txt'), np.array([test_acc]))
    save_matrix(os.path.join(folder, 'train_time.txt'), np.array([train_time, test_time]))

for dataset in datasets:
    results_dir = os.path.join(RESULTS_DIR, dataset['name'])
    models_dir = os.path.join(MODELS_DIR, dataset['name'])
    create_dir(results_dir)
    create_dir(models_dir)
    train_provider, val_provider, test_provider = load_dataset(dataset)
    for method in methods:
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
        method_results_dir = os.path.join(results_dir, method['name'] + '_' + timestamp)
        method_model_dir = os.path.join(models_dir, method['name'] + '_' + timestamp)
        create_dir(method_results_dir)
        create_dir(method_model_dir)

        start_time = time()
        classifier, hist = train(method, train_provider, val_provider)
        train_time = (time()-start_time) / num_epochs

        start_time = time()
        conf_mat, test_acc = test(classifier, test_provider, method=method, dataset=dataset)
        test_time = (time() - start_time) / num_epochs

        save_results(method_results_dir, hist, conf_mat, test_acc, train_time, test_time)
        if SAVE_MODELS:
            save_model(method_model_dir, classifier)



# tensorboard = TensorBoard(log_dir=log_dir+'log/{}'.format(time()), write_graph=False)
# tensorboard = TensorBoard(log_dir=os.path.join(log_dir, 'log'))



