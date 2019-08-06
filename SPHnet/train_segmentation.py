import os

"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
from utils.pointclouds_utils import rotate_point_cloud_batch
from data_providers.seg_provider import SegmentationProvider
import keras
import keras.backend as K
from keras.models import load_model
from keras.metrics import sparse_categorical_accuracy, categorical_accuracy
from time import time
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.save_model import save_matrix, create_dir, save_training_acc, save_model
from methods.segmentation_methods import methods_list
from data_providers.segmentation_datasets import datasets_list
from utils.pointclouds_utils import pc_batch_preprocess
import h5py

datasets = datasets_list
methods = methods_list

MODELS_DIR = 'C:/Users/adrien/Documents/models'
RESULTS_DIR = 'C:/Users/adrien/Documents/results'
PRED_DIR = 'C:/Users/adrien/Documents/preds'
# WEIGHTS_PATH = os.path.join(MODELS_DIR, 'C:/Users/adrien/Documents/models/dfaust_matching_augmented/btree_inv_conv_2019_06_06_13_40_12', 'weights.h5')
WEIGHTS_PATH = None

assert(os.path.isdir(MODELS_DIR))
assert(os.path.isdir(RESULTS_DIR))
SAVE_MODELS = True
SAVE_PREDS = True

num_points = 2048
batch_size = 8
num_epochs = 100
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

    num_parts = dataset['num_parts']
    num_classes = dataset['num_classes']
    parts = dataset['parts']

    # cat_to_labels = dataset['cat_to_labels']
    labels_to_cat = dataset['labels_to_cat']

    train_provider = SegmentationProvider(files_list=train_files_list,
                                          data_path=train_data_folder,
                                          n_parts=num_parts,
                                          n_classes=num_classes,
                                          n_points=num_points,
                                          batch_size=batch_size,
                                          preprocess=train_preprocessing,
                                          shuffle=SHUFFLE,
                                          parts=parts)

    val_provider = SegmentationProvider(files_list=val_files_list,
                                        data_path=val_data_folder,
                                        n_parts=num_parts,
                                        n_classes=num_classes,
                                        n_points=num_points,
                                        batch_size=batch_size,
                                        preprocess=val_preprocessing,
                                        shuffle=SHUFFLE,
                                        parts=parts)

    test_provider = SegmentationProvider(files_list=test_files_list,
                                         data_path=test_data_folder,
                                         n_parts=num_parts,
                                         n_classes=num_classes,
                                         n_points=num_points,
                                         batch_size=batch_size,
                                         preprocess=test_preprocessing,
                                         shuffle=False,
                                         parts=parts,
                                         labels_to_cat=labels_to_cat)

    return train_provider, val_provider, test_provider


def train(method, train_provider, val_provider):
    num_parts = train_provider.n_parts
    num_categories = train_provider.n_classes
    # segmenter = SegNetwork(method=method)

    segmenter = method['arch'](method=method)
    segmenter = segmenter.get_network_model(part_num=num_parts,
                                            num_points=num_points,
                                            num_categories=num_categories,
                                            batch_size=batch_size)

    def acc(y_true, y_pred):
        # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
        if K.ndim(y_true) == K.ndim(y_pred):
            y_true = K.squeeze(y_true, -1)
        # convert dense predictions to labels
        y_pred_labels = K.argmax(y_pred, axis=-1)
        y_pred_labels = K.cast(y_pred_labels, K.floatx())
        return K.cast(K.equal(y_true, y_pred_labels), K.floatx())



    # loss_weights = train_provider.part_weights
    metric = acc
    segmenter.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer='adam',
                      metrics=[metric])

    segmenter.summary()

    if WEIGHTS_PATH is None:
        hist = segmenter.fit_generator(generator=train_provider,
                                       validation_data=val_provider,
                                       epochs=num_epochs,
                                       callbacks=[],
                                       verbose=2)
    else:
        segmenter.load_weights(WEIGHTS_PATH)
        hist = 0
        
    return segmenter, hist

def IoU__(one_hot_pred, one_hot, num_parts):
    ones = np.ones(shape=(num_parts, ))
    zeros = np.zeros(shape=(num_parts, ))
    true_sum = np.sum(one_hot, axis=0, keepdims=False)
    parts = np.where(true_sum > 0, ones, zeros)
    num_parts = np.sum(parts)
    Inter = np.multiply(one_hot_pred, one_hot)
    Inter = np.sum(Inter, axis=0, keepdims=False)
    Union = np.maximum(one_hot_pred, one_hot)
    Union = np.sum(Union, axis=0, keepdims=False)

    Union = np.where(Union == 0, ones, Union)

    IoU = np.divide(Inter, Union)

    """
    print('parts ', parts)
    print('num_parts ', num_parts)
    print('union ', Union)
    print('inter ', Inter)
    print('iou ', IoU)
    """

    IoU = np.sum(IoU) / num_parts

    return IoU


def save_part_labels(path, name, pred, true, data):
    h5_filename = os.path.join(path, name + '.hdf5')
    h5_fout = h5py.File(h5_filename)

    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype='float32')

    h5_fout.create_dataset(
        'pred', data=pred,
        compression='gzip', compression_opts=1,
        dtype='uint8')

    h5_fout.create_dataset(
        'true', data=true,
        compression='gzip', compression_opts=1,
        dtype='uint8')

    h5_fout.close()


def test(segmenter, test_provider, method=None, dataset=None):
    data, part_labels, class_labels = test_provider.get_data()
    batch_size_ = test_provider.get_batch_size()
    # extend data by batch size

    cat_to_labels = test_provider.cat_to_labels
    # labels_to_cat = test_provider.labels_to_cat

    seg_parts = test_provider.seg_parts

    shape_ious = {cat: [] for cat in cat_to_labels.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_parts.keys():

        for label in seg_parts[cat]:
            seg_label_to_cat[label] = cat

    data = np.concatenate([data, data[:batch_size_, ...]], axis=0)
    part_labels = np.concatenate([part_labels, part_labels[:batch_size_, ...]], axis=0)
    class_labels = np.concatenate([class_labels, class_labels[:batch_size_, ...]], axis=0)



    num_samples = data.shape[0]
    num_batches = num_samples // batch_size_
    num_parts = test_provider.get_num_parts()
    num_classes = test_provider.get_num_classes()
    num_points = data.shape[1]
    num_points_target = test_provider.get_num_points()
    preprocess = test_provider.get_preprocess()


    pred_part_labels = np.zeros(shape=(num_samples, num_points_target), dtype=np.int32)
    part_labels_ = np.zeros(shape=(num_samples, num_points_target), dtype=np.int32)
    data_ = np.zeros(shape=(num_samples, num_points_target, 3), dtype=np.float32)

    acc = 0.
    mIoU = 0.
    per_class_iou = np.zeros((num_classes, ))
    total_seen_per_cat = np.zeros((num_classes, ), dtype=np.float32)
    total_seg_acc_per_cat = np.zeros((num_classes, ), dtype=np.float32)
    for i in range(num_batches):




        idx = np.random.permutation(np.arange(num_points))[:num_points_target]
        cur_data = data[i*batch_size_:(i+1)*batch_size_, ...]
        cur_data = cur_data[:, idx, ...]
        cur_part_labels = part_labels[i*batch_size_:(i+1)*batch_size_, ...]
        cur_part_labels = cur_part_labels[:, idx, ...]

        # kd tree idx
        m = []
        for j in range(len(preprocess)):
            if preprocess[j] == 'rotate':
                x, m = rotate_point_cloud_batch(cur_data)
                cur_data = x
            else:
                cur_data, cur_part_labels = pc_batch_preprocess(cur_data, y=cur_part_labels, proc=preprocess[j])


        cur_one_hot_part_labels = keras.utils.to_categorical(cur_part_labels, num_classes=num_parts)
        cur_part_labels = cur_part_labels[..., 0]

        part_labels_[i*batch_size_:(i+1)*batch_size_, ...] = cur_part_labels

        d_ = cur_data.copy()
        for k in range(len(m)):
            d_[k, ...] = np.dot(cur_data[k, ...], m[k].T)

        data_[i * batch_size_:(i + 1) * batch_size_, ...] = d_.copy()

        cur_class_labels = class_labels[i*batch_size_:(i+1)*batch_size_, ...]
        cur_one_hot_class_labels = keras.utils.to_categorical(cur_class_labels, num_classes=num_classes)
        cur_pred_part_labels = segmenter.predict_on_batch(x=[cur_data, cur_one_hot_class_labels])
        cur_pred_part_labels = np.argmax(cur_pred_part_labels, axis=-1)

        pred_part_labels[i*batch_size_:(i+1)*batch_size_, ...] = cur_pred_part_labels

        acc_ = np.equal(cur_part_labels, cur_pred_part_labels)
        acc_ = acc_.astype(np.float)
        per_instance_part_acc = np.mean(acc_, axis=1)
        acc += np.mean(per_instance_part_acc)

        for j in range(batch_size_):

            total_seen_per_cat[cur_class_labels[j]] += 1.
            total_seg_acc_per_cat[cur_class_labels[j]] += per_instance_part_acc[j]


            segp = cur_pred_part_labels[j, :]
            segl = cur_part_labels[j, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_parts[cat]))]
            for l in seg_parts[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_parts[cat][0]] = 1.0
                else:
                    part_ious[l - seg_parts[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    k = 0
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
        per_class_iou[k] = shape_ious[cat]
        k += 1

    mIoU = np.mean(all_shape_ious)
    mean_class_iou = np.mean(np.array(list(shape_ious.values())))

    acc /= num_batches
    per_class_acc = np.divide(total_seg_acc_per_cat, total_seen_per_cat)
    mean_class_acc = np.mean(per_class_acc)


    if method is not None and dataset is not None:
        print(method['name'] + ' test_acc on ' + dataset['name'] + ' dataset is ', acc)

    part_labels_ = np.reshape(part_labels_, newshape=(-1, ))
    pred_part_labels = np.reshape(pred_part_labels, newshape=(-1, ))

    conf_mat = confusion_matrix(part_labels_, pred_part_labels)

    return acc, per_class_acc, mean_class_acc, mIoU, per_class_iou, mean_class_iou, conf_mat, data_, part_labels_, \
           pred_part_labels

def save_model_(dir, method, model, timestamp):
    folder = os.path.join(dir, method['name'] + '_' + timestamp)
    save_model(model, folder)

def save_results_(dir, hist, conf_mat, test_acc, train_time, test_time):
    folder = dir
    save_matrix(os.path.join(folder, 'confusion_matrix.txt'), conf_mat)
    save_training_acc(folder, hist)
    save_matrix(os.path.join(folder, 'test_acc.txt'), np.array([test_acc]))
    save_matrix(os.path.join(folder, 'train_time.txt'), np.array([train_time, test_time]))

def save_train_results(dir, hist, train_time):
    folder = dir
    save_training_acc(folder, hist)
    save_matrix(os.path.join(folder, 'train_time.txt'), np.array([train_time]))

def save_test_results(dir, acc, per_class_acc, mean_class_acc,
                      mIoU, per_class_iou, mean_class_iou,
                      conf_mat, test_time):

    save_matrix(os.path.join(dir, 'test_acc.txt'), np.array([acc]))
    save_matrix(os.path.join(dir, 'per_class_acc.txt'), per_class_acc)
    save_matrix(os.path.join(dir, 'mean_class_acc.txt'), np.array([mean_class_acc]))

    save_matrix(os.path.join(dir, 'mIoU.txt'), np.array([mIoU]))
    save_matrix(os.path.join(dir, 'per_class_iou.txt'), per_class_iou)
    save_matrix(os.path.join(dir, 'mean_class_iou.txt'), np.array([mean_class_iou]))

    save_matrix(os.path.join(dir, 'confusion_matrix.txt'), conf_mat)
    save_matrix(os.path.join(dir, 'test_time.txt'), np.array([test_time]))

for dataset in datasets:
    results_dir = os.path.join(RESULTS_DIR, dataset['name'])
    models_dir = os.path.join(MODELS_DIR, dataset['name'])
    pred_dir = os.path.join(PRED_DIR, dataset['name'])
    create_dir(results_dir)
    if SAVE_MODELS:
        create_dir(models_dir)
    if SAVE_PREDS:
        create_dir(pred_dir)
    train_provider, val_provider, test_provider = load_dataset(dataset)
    for method in methods:
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
        method_results_dir = os.path.join(results_dir, method['name'] + '_' + timestamp)
        method_model_dir = os.path.join(models_dir, method['name'] + '_' + timestamp)
        method_pred_dir = os.path.join(pred_dir, method['name'] + '_' + timestamp)
        create_dir(method_results_dir)

        if SAVE_MODELS:
            create_dir(method_model_dir)

        if SAVE_PREDS:
            create_dir(method_pred_dir)

        start_time = time()
        classifier, hist = train(method, train_provider, val_provider)
        train_time = (time()-start_time) / num_epochs

        # save_train_results(method_results_dir, hist, train_time)

        start_time = time()
        acc, per_class_acc, mean_class_acc, mIoU, per_class_iou, mean_class_iou, conf_mat, data_, part_labels_, \
           pred_part_labels_ = \
            test(classifier, test_provider, method=method, dataset=dataset)
        test_time = (time() - start_time)

        save_test_results(method_results_dir,
                          acc, per_class_acc, mean_class_acc,
                          mIoU, per_class_iou, mean_class_iou, conf_mat, test_time)

        # save_results(method_results_dir, hist, conf_mat, test_acc, train_time, test_time)

        if SAVE_PREDS:
            save_part_labels(method_pred_dir, dataset['name'] + '_' + method['name'],
                             pred_part_labels_, part_labels_, data_)

        if SAVE_MODELS:
            save_model(method_model_dir, classifier)