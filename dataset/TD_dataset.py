import numpy as np
import torch
import glob
import os
import csv
from torch.utils.data import Dataset


class RF_Time_Dataset(Dataset):
    r'''
    Basic class for RF time domain dataset
    
    Args:
        root_dir: str, path to the root directory of the dataset
        crop: tuple, targeted time domain data range in format (lower distance limit, high distance limit)
        dataset_config: dict, dictionary containing the split settings for either training or validation datasets
        
    Attributes:
        root_dir: str, path to the root directory of the dataset
        data_list: list, list of paths to all csv files in the dataset
        category: dict, dictionary mapping category names to indices
    
    Methods:
        __len__: returns the length of the dataset
        
        __getitem__: returns the complex data and its label at the given index
        
        standardize_complex_whiten: standardizes the complex data through data whitening,
        the processed complex data has zero mean and unit variance (covariance matrix = identity matrix)
        
        load_raw_data: loads the raw data from the csv file
        
        load_thz_data: loads the complex data from the csv file
    '''
    def __init__(self, root_dir: str, dataset_config: dict = None, split_mode: str = 'random_split'):
        self.root_dir = root_dir
        # check if the dataset_config is valid
        if dataset_config and 'data_form' not in dataset_config:
            raise ValueError(
                "dataset_config must contain 'data_form' key for this data version. "
                "Example format: {'data_form': {'distances': ['20', '30']}}"
            )
        self.dataset_config = dataset_config
        self.split_mode = split_mode
        self.data_list, self.class_mapping = self._get_data()
        
    def _get_data_info(self, temp_folder_dir):
        r'''
        Get the distance information from the folder path
        
        Args:
            temp_folder_dir: str, path to each folder in the dataset. 
            e.g. '~/D:/Data/thz/data/ms46131a-4To43.5GHz-2048-brick01-200mm-0deg'
            
        returns:
            distance: int, distance information extracted from the path. units in 'mm'
            e.g. '200'
            degree: int, angle information extracted from the path. units in 'deg'
            e.g. '0'
            class_name: str, class name information extracted from the path.
        '''
        folder_name = os.path.basename(temp_folder_dir)
        distance = int(folder_name.split('-')[-2].split('mm')[0])  # e.g. '200'
        degree = int(folder_name.split('-')[-1].split('deg')[0]) # e.g. '0'
        class_name = folder_name.split('-')[-3]  # e.g. 'brick01'
        return distance, degree, class_name
        
    def _get_data(self):
        r'''
        Get data list and class_mapping dictionary for different data versions
        
        Args:
            data_config: dict, dictionary containing the split settings for either training or validation datasets. If None, all data will be used.
            
        Returns:
            data_list: list, list of paths to all csv files in the dataset
            class_mapping: dict, dictionary mapping class names to indices
        '''
        # data_list = glob.glob(os.path.join(self.root_dir, '*/*.csv'))
        folder_list = glob.glob(os.path.join(self.root_dir, '*'))
        # print(folder_list)
        # FIRST: generate class_mapping dictionary based on the folder names
        class_names = {os.path.basename(path).split('-')[-3] for path in folder_list}
        # class_names = {'cement1', 'cardboard1', 'cement2', 'leather1', 'metal1', 'ceramicTile1'}
        class_mapping = {class_name: i for i, class_name in enumerate(sorted(class_names))}
        # Second: generate data_list, if dataset_config != None, filtering with train or val setting
        data_list = []
        if self.split_mode == 'random_split':
            # If split_mode is 'random_split', use all data
            for temp_folder_dir in folder_list:
                temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                data_list.extend(temp_data_list)
        elif self.split_mode == 'cross_distance_split':
            # If split_mode is 'cross_distance_split', filter data based on the dataset_config
            for temp_folder_dir in folder_list:
                distance, degree, class_name = self._get_data_info(temp_folder_dir)
                if distance in self.dataset_config['data_form']['distances']:
                    temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend(temp_data_list)
        elif self.split_mode == 'cross_angle_split':
            # If split_mode is 'cross_angle_split', filter data based on the dataset_config
            for temp_folder_dir in folder_list:
                distance, degree, class_name = self._get_data_info(temp_folder_dir)
                if degree in self.dataset_config['data_form']['angles']:
                    temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend(temp_data_list)
        else:
            raise ValueError("split_mode must be either 'random_split', 'cross_distance_split' or 'cross_angle_split'")
            
        return data_list, class_mapping
        
    def __len__(self):
        r'''Returns the length of the dataset'''
        return len(self.data_list)
    
    def __getitem__(self, idx):
        r'''Returns the complex data and its label at the given index'''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        temp_dir = self.data_list[idx]

        label_name = os.path.basename(os.path.dirname(temp_dir)).split('-')[-3]
        y = self.class_mapping[label_name]

        x = self.load_thz_data(temp_dir)
        return x, y
    
    def normalize(self, x):
        r'''Conducts data normalization on the Time Domain 1D data'''
        return (x - torch.mean(x)) / torch.std(x)

    def load_raw_data(self, temp_dir):
        r'''load the time-domain data from the csv file'''
        temp_data = []
        with open(temp_dir, 'r') as temp_file:
            reader = csv.reader(temp_file)
            for row in reader:
                temp_data.append(row)
        temp_data = np.array(temp_data[1:], dtype=np.float32)
        return temp_data  

    def load_thz_data(self, temp_dir):
        r'''
        loads the complex data from the csv file
        
        Args:
            temp_dir: str, path to the csv file
        
        Returns:
            complex_x: torch.complex, 1D tensor with shape (1024,) representing the whitened complex data
        '''
        temp_data = self.load_raw_data(temp_dir)
        time_position = torch.FloatTensor(temp_data[:, 0])  # in ps
        amplitude = torch.FloatTensor(temp_data[:, 1])
        amplitude = amplitude[::3]
        norm_amplitude = self.normalize(amplitude)
        # return torch.cat((time_position.unsqueeze(-1), norm_amplitude.unsqueeze(-1)), dim=-1)  # shape (10240, 2)
        return norm_amplitude.unsqueeze(-1)