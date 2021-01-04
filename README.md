# [SPHNet](https://arxiv.org/abs/1906.11555)
This is our implementation of SPHNet, a rotation invariant deep learning architecture for point clouds analysis.


## Prerequisites
* CUDA and CuDNN
* Python >= 3.5
* Tensorflow 1.8
* Keras

## How to train ?

The code proposes two settings: classification and segmentation

# Classification
The file classification_dataset.py in the data_providers folder allows you to specify a dataset for shape classification as a dictionary containing:
* 'name' : a name for the dataset
* 'num_classes': the number of classes
* 'train_data_folder': path of training data folder
* 'val_data_folder': path of validation data folder
* 'test_data_folder': path of test data folder
* 'train_files_list': path of a txt file containing a list of training hdf5 files
* 'val_files_list': path of a txt file containing a list of validation hdf5 files
* 'test_files_list': path of a txt file containing a list of test hdf5 files,
* 'train_preprocessing': a list of preprocessing functions to be applied on each training batch before sending it to the GPU
* 'val_preprocessing': a list of preprocessing functions to be applied on each training batch before sending it to the GPU
* 'test_preprocessing': a list of preprocessing functions to be applied on each test batch before sending it to the GPU

Preprocessing functions can be found in utils/pointclouds_utils.py, (random scaling / rotation, kd_tree indexing ...).
Finally add the dataset dictionary to the datasets list.

Indicate a path for saving the results (RESULTS_DIR) and models (MODELS_DIR) in train_classification.py and run the script to 
train the network.

# Segmentation 
Segmentation datasets are specified similarly in data_providers/segmentation_datasets.py 
you need to specify the number of parts. You can specify a path to save the results and trained models as well as the predicted 
labels for the test set (PRED_DIR), run the script to train the network on the segmentation dataset.


# Data

* You can download the ModelNet40 dataset for classification at : https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip



## Citation
If you use our work, please cite our paper.
```
@article{poulenard2019effective,
  title={Effective Rotation-invariant Point CNN with Spherical Harmonics kernels},
  author={Poulenard, Adrien and Rakotosaona, Marie-Julie and Ponty, Yann and Ovsjanikov, Maks},
  journal={arXiv preprint arXiv:1906.11555},
  year={2019}
}
```
