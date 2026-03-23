from matplotlib.dates import TH
import numpy as np
import torch
import glob
import os
import csv
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class RFMatID_Freq_Dataset(Dataset):
    r'''
    RF-MatID frequency domain dataset

    Attributes:
        root_dir: str, path to the root directory of the dataset
        dataset_config: dict, dictionary containing the split settings for either training or validation datasets
        split_mode: str, mode of splitting the dataset ('random_split' or 'cross_distance_split' or 'cross_angle_split')
        crop: list of tuples, each tuple represents a targeted frequency domain data range in format (lower feq limit, high feq limit),
        value is chosen from 4. to 43.5 GHz, e.g. [(4., 15.), (20., 30.)]
        freq_data_type: str, choose from ['complex', 'dual_channel', 'three_channel], the type of frequency domain data to use

    Methods:
        __len__: returns the length of the dataset
        __getitem__: returns the complex data and its label at the given index
        _get_data_info: gets the distance, angle and class information from the folder path
        _get_data: gets the data list and class_mapping dictionary for different data versions
        standardize_complex_whiten: standardizes the complex data through data whitening,
        the processed complex data has zero mean and unit variance (covariance matrix = identity matrix)
        load_raw_data: loads the raw data from the csv file
        freq_crop: crop the trageted frequency domain data from loaded raw complex data
        load_thz_data: loads the complex data from the csv file
    '''

    def __init__(self, root_dir: str, crop: tuple = None, dataset_config: dict = None, freq_data_type: str = 'dual_channel', split_mode: str = 'random_split', application_setting: str = 'all_classes'):
        r'''
        Args:
            root_dir: str, path to the root directory of the dataset
            crop: list of tuples, each tuple represents a targeted frequency domain data range in format (lower feq limit, high feq limit), value is chosen from 4. to 43.5 GHz, e.g. [(4., 15.), (20., 30.)]
            dataset_config: dict, dictionary containing the split settings for either training or validation datasets
            freq_data_type: str, choose from ['complex', 'dual_channel', 'three_channel'], the type of frequency domain data to use
            split_mode: str, mode of splitting the dataset ('random_split' or 'cross_distance_split' or 'cross_angle_split')
            application_setting: str, the application setting for the dataset, choose from ['all_classes', 'super_classes'], default is 'all_classes'. If 'super_classes' is chosen, the dataset will be filtered to only include coarse-grained classes (e.g., brick, stone, wood, syntheticMaterial, glass) instead of fine-grained classes (e.g., brick01, brick02, stone01, etc.)
        '''
        self.root_dir = root_dir
        # check if the dataset_config is valid
        if dataset_config and 'data_form' not in dataset_config:
            raise ValueError(
                "dataset_config must contain 'data_form' key for this data version. "
                "Example format: {'data_form': {'distances': ['20', '30']}}"
            )
        self.dataset_config = dataset_config
        self.split_mode = split_mode
        self.application_setting = application_setting
        if application_setting == 'all_classes':
            self.data_list, self.class_mapping, self.data_labels = self._get_data()
        elif application_setting == 'super_classes':
            self.data_list, self.class_mapping, self.data_labels = self._get_data_super_class()
        else:
            raise ValueError("application_setting must be either 'all_classes' or 'super_classes'")
        self.crop = crop
        self.freq_data_type = freq_data_type

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
        degree = int(folder_name.split('-')[-1].split('deg')[0])  # e.g. '0'
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
        # FIRST: generate class_mapping dictionary based on the folder names
        class_names = {os.path.basename(path).split(
            '-')[-3] for path in folder_list}
        # class_names = {'cement1', 'cardboard1', 'cement2', 'leather1', 'metal1', 'ceramicTile1'}
        class_mapping = {class_name: i for i,
                         class_name in enumerate(sorted(class_names))}
        # Second: generate data_list, if dataset_config != None, filtering with train or val setting
        data_list = []
        if self.split_mode == 'random_split':
            # If split_mode is 'random_split', use all data
            for temp_folder_dir in folder_list:
                temp_data_list = glob.glob(
                    os.path.join(temp_folder_dir, '*.csv'))
                data_list.extend(temp_data_list)
        elif self.split_mode == 'cross_distance_split':
            # If split_mode is 'cross_distance_split', filter data based on the dataset_config
            for temp_folder_dir in folder_list:
                distance, degree, class_name = self._get_data_info(
                    temp_folder_dir)
                if distance in self.dataset_config['data_form']['distances']:
                    temp_data_list = glob.glob(
                        os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend(temp_data_list)
        elif self.split_mode == 'cross_angle_split':
            # If split_mode is 'cross_angle_split', filter data based on the dataset_config
            for temp_folder_dir in folder_list:
                distance, degree, class_name = self._get_data_info(
                    temp_folder_dir)
                if degree in self.dataset_config['data_form']['angles']:
                    temp_data_list = glob.glob(
                        os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend(temp_data_list)
        else:
            raise ValueError(
                "split_mode must be either 'random_split', 'cross_distance_split' or 'cross_angle_split'")
        data_labels = [os.path.basename(os.path.dirname(path)).split('-')[-3] for path in data_list]  # Save the class labels (strings)
       
        return data_list, class_mapping, data_labels
    
    def _coarse_class(self, raw_class: str) -> str:
        coarse_classes = ['brick', 'stone', 'wood', 'syntheticMaterial', 'glass']
        raw_lower = raw_class.lower()
        for c in coarse_classes:
            if raw_lower.startswith(c.lower()):  # Convert both sides to lowercase
                return c  # Return to the original hump form (keep the class mapping)
        raise ValueError(f"Unknown class prefix: {raw_class}")

    def _get_data_super_class(self):
        r'''
        Get data list and class_mapping dictionary for different data versions
        Now maps fine-grained classes (e.g., brick01) to coarse classes (e.g., brick)
        
        Returns:
            data_list: list, list of paths to all csv files in the dataset
            class_mapping: dict, mapping from coarse class name to index (0~4)
        '''
        folder_list = glob.glob(os.path.join(self.root_dir, '*'))

        # First, get all coarse-grained categories
        coarse_class_names = set()
        folder_to_class = {}  # Cache the coarse categories corresponding to each folder

        for path in folder_list:
            folder_name = os.path.basename(path)
            raw_class = folder_name.split('-')[-3]  # e.g., 'brick01'
            try:
                coarse_class = self._coarse_class(raw_class)
                coarse_class_names.add(coarse_class)
                folder_to_class[path] = coarse_class
            except ValueError:
                print(f"Skipping unknown class: {raw_class}")
                continue

        # Alphabetical order ensures that the mapping is fixed
        sorted_classes = sorted(coarse_class_names)
        class_mapping = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}

        # Second, build data_list and record labels (using coarse categories) 
        data_list = []

        if self.split_mode == 'random_split':
            for temp_folder_dir in folder_list:
                coarse_class = folder_to_class.get(temp_folder_dir)
                if coarse_class is None:
                    continue
                temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                data_list.extend([(p, coarse_class) for p in temp_data_list])  # Save the path + tag

        elif self.split_mode == 'cross_distance_split':
            for temp_folder_dir in folder_list:
                distance, degree, raw_class = self._get_data_info(temp_folder_dir)
                coarse_class = folder_to_class.get(temp_folder_dir)
                if coarse_class is None:
                    continue
                if distance in self.dataset_config['data_form']['distances']:
                    temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend([(p, coarse_class) for p in temp_data_list])

        elif self.split_mode == 'cross_angle_split':
            for temp_folder_dir in folder_list:
                distance, degree, raw_class = self._get_data_info(temp_folder_dir)
                coarse_class = folder_to_class.get(temp_folder_dir)
                if coarse_class is None:
                    continue
                if degree in self.dataset_config['data_form']['angles']:
                    temp_data_list = glob.glob(os.path.join(temp_folder_dir, '*.csv'))
                    data_list.extend([(p, coarse_class) for p in temp_data_list])
        else:
            raise ValueError("split_mode must be either 'random_split', 'cross_distance_split' or 'cross_angle_split'")

        # Third, separate paths and labels
        data_labels = [item[1] for item in data_list]  # Save labels (strings)
        data_list = [item[0] for item in data_list]         # Only save the paths

        return data_list, class_mapping, data_labels

    def __len__(self):
        r'''Returns the length of the dataset'''
        return len(self.data_list)

    def __getitem__(self, idx):
        r'''Returns the complex data and its label at the given index'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        temp_dir = self.data_list[idx]
        label_name = self.data_labels[idx]
        y = self.class_mapping[label_name]

        x = self.load_thz_data(temp_dir)
        return x, y

    def standardize_complex_whiten(self, real_data: torch.FloatTensor, imag_data: torch.FloatTensor):
        r'''
        Standardizes the complex data through data whitening,
        the processed complex data has zero mean and unit variance (covariance matrix = identity matrix)

        Args:
            real_data: torch.FloatTensor, 1D tensor with shape (1024,) representing the real part of the complex data
            imag_data: torch.FloatTensor, 1D tensor with shape (1024,) representing the imaginary part of the complex data

        Returns:
            whitened_data: torch.FloatTensor, 2D tensor with shape (1024, 2) representing the whitened complex data
        '''
        # real_data and imag_data are 1D tensors with shape (1024,)
        combined_data = torch.concatenate(
            (real_data.unsqueeze(-1), imag_data.unsqueeze(-1)), axis=-1)
        # combined_data is a 2D tensor with shape (1024, 2)
        # Compute mean and covariance
        mean = torch.mean(combined_data, dim=0)
        centered = combined_data - mean
        cov = torch.mm(centered.T, centered) / (centered.shape[0] - 1)
        # Compute inverse square root of covariance matrix
        eigvals, eigvecs = torch.linalg.eigh(cov)
        sqrt_inv_cov = eigvecs @ torch.diag(1 /
                                            torch.sqrt(eigvals)) @ eigvecs.T
        # Whiten the data
        whitened_data = (centered @ sqrt_inv_cov.T).view_as(combined_data)
        return whitened_data

    def load_raw_data(self, temp_dir):
        r'''
        loads the raw data from the csv file

        Args:
            temp_dir: str, path to the csv file

        Returns:
            temp_data: np.array, 2D array with shape (1024, 3), columns are ['freq.GHZ', 'real', 'imag']
        '''
        temp_data = []
        with open(temp_dir, 'r') as temp_file:
            reader = csv.reader(temp_file)
            for row in reader:
                temp_data.append(row)
        temp_data = np.array(temp_data[1:], dtype=np.float32)
        return temp_data

    def freq_crop(self, raw_complex_data):
        r'''
        Crop the trageted frequency domain data from loaded raw complex data

        Args:
            raw_complex_data: np.array, 2D array with shape (1024, 3), columns are ['freq.GHZ', 'real', 'imag']

        Returns:
            raw_complex_data: np.array, 2D array with shape (freq_len, 3), columns are ['freq.GHZ', 'real', 'imag']
        '''

        cropped_data = np.empty((0, 3), dtype=float)
        for crop_range in self.crop:
            if not (4. <= crop_range[0] < crop_range[1] <= 43.5):
                raise ValueError("Crop range must be within [4., 43.5] GHz")
            else:
                temp_cropped_data = raw_complex_data[(
                    raw_complex_data[:, 0] >= crop_range[0]*1e9) & (raw_complex_data[:, 0] <= crop_range[1]*1e9)]
                cropped_data = np.vstack((cropped_data, temp_cropped_data))
        return cropped_data

    def load_thz_data(self, temp_dir):
        r'''
        loads the complex data from the csv file

        Args:
            temp_dir: str, path to the csv file

        Returns:
            complex_x: torch.complex, 1D tensor with shape (1024,) representing the whitened complex data
        '''
        temp_data = self.load_raw_data(
            temp_dir)  # 2D array with shape (1024, 3), columns are ['freq.GHZ', 'real', 'imag']
        if self.crop is not None:
            temp_data = self.freq_crop(temp_data)
        freq_position = torch.FloatTensor(temp_data[:, 0]) / 1e9
        temp_real_data, temp_imag_data = torch.FloatTensor(
            temp_data[:, 1]), torch.FloatTensor(temp_data[:, 2])
        whitened_data = self.standardize_complex_whiten(
            temp_real_data, temp_imag_data)
        if self.freq_data_type == 'complex':
            complex_x = torch.complex(whitened_data[:, 0], whitened_data[:, 1])
            return complex_x
        elif self.freq_data_type == 'dual_channel':
            return whitened_data
        elif self.freq_data_type == 'three_channel':
            return torch.cat((freq_position.unsqueeze(-1), whitened_data), dim=-1)
        else:
            raise ValueError(
                "freq_data_type must be chosen from 'complex', 'dual_channel', or 'three_channel'")
