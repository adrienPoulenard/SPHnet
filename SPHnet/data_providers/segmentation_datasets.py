import os


dataset_dir = './Datasets/RF00001-family/hdf5_data'

train_files_list = os.path.join(dataset_dir, 'train_hdf5_file_list.txt')
val_files_list = os.path.join(dataset_dir, 'val_hdf5_file_list.txt')
test_files_list = os.path.join(dataset_dir, 'test_hdf5_file_list.txt')


train_data_folder = dataset_dir
val_data_folder = dataset_dir
test_data_folder = dataset_dir


rna_seg_augmented = {'name': 'rna_seg_augmented_test_rotated',
                     'num_parts': 160,
                     'num_classes': 1,
                     'parts': [],
                     'labels_to_cat': None,
                     'train_data_folder': train_data_folder,
                     'val_data_folder': val_data_folder,
                     'test_data_folder': test_data_folder,
                     'train_files_list': train_files_list,
                     'val_files_list': val_files_list,
                     'test_files_list': test_files_list,
                     'train_preprocessing': ['rotate', 'scale', 'kd_tree_idx'],
                     'val_preprocessing': ['kd_tree_idx'],
                     'test_preprocessing': ['rotate', 'kd_tree_idx']}

rna_seg = {'name': 'rna_seg_test_rotated',
                    'num_parts': 160,
                    'num_classes': 1,
                    'parts': [],
                    'labels_to_cat': None,
                    'train_data_folder': train_data_folder,
                    'val_data_folder': val_data_folder,
                    'test_data_folder': test_data_folder,
                    'train_files_list': train_files_list,
                    'val_files_list': val_files_list,
                    'test_files_list': test_files_list,
                    'train_preprocessing': ['scale', 'kd_tree_idx'],
                    'val_preprocessing': ['kd_tree_idx'],
                    'test_preprocessing': ['rotate', 'kd_tree_idx']}


datasets_list = [rna_seg, rna_seg_augmented]
