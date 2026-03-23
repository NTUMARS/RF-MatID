import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import glob
import os
import csv


class Split_Config_Gen:
    def get_cross_distance_config(self, mod):
        all_dists = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                     1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600,
                     1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
        
        if mod == 'mod1':
            val = [250, 400, 550, 700, 900, 1050, 1200, 1350, 1550, 1700, 1900]
        elif mod == 'mod2':
            val = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        elif mod == 'mod3':
            val = [1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
        else: val = []
        
        train = [d for d in all_dists if d not in val]
        return {'train': {'distances': train}, 'val': {'distances': val}}

    def get_cross_angle_config(self, mod):
        all_angs = list(range(11)) # 0-10
        if mod == 'mod1': val = [2, 5, 8]
        elif mod == 'mod2': val = [0, 1, 2]
        elif mod == 'mod3': val = [8, 9, 10]
        else: val = []
        
        train = [a for a in all_angs if a not in val]
        return {'train': {'angles': train}, 'val': {'angles': val}}

class RFVibeS11Dataset(Dataset):
    def __init__(self, root_dir, crop=None, split_config=None, split_mode='random_split'):
        """
        Args:
            root_dir: Data path
            crop: Frequency range tuple, e.g., (4.0, 30.0). None for all range.
            split_config: A dict from Split_Generator 
            split_mode: 'random_split', 'cross_distance_split', 'cross_angle_split'
        """
        self.root_dir = root_dir
        self.crop = crop
        self.data_list = []
        self.labels = []
        
        all_folders = glob.glob(os.path.join(root_dir, '*'))
        
        class_names = sorted(list({os.path.basename(p).split('-')[-3] for p in all_folders}))
        self.class_mapping = {name: i for i, name in enumerate(class_names)}
        
        for folder in all_folders:
            fname = os.path.basename(folder)
            parts = fname.split('-')
            cls_name = parts[-3]
            dist = int(parts[-2].replace('mm',''))
            deg = int(parts[-1].replace('deg',''))

            keep = False
            if split_mode == 'random_split':
                keep = True 
            elif split_mode == 'cross_distance_split':
                if dist in split_config['distances']: keep = True
            elif split_mode == 'cross_angle_split':
                if deg in split_config['angles']: keep = True
            
            if keep:
                csvs = glob.glob(os.path.join(folder, '*.csv'))
                for c in csvs:
                    self.data_list.append(c)
                    self.labels.append(self.class_mapping[cls_name])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fpath = self.data_list[idx]
        label = self.labels[idx]
        
        feat = self._process_single_file(fpath)

        
        return (
                torch.tensor(feat[0], dtype=torch.float32), 
                torch.tensor(feat[1], dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long)
            )
    
    def _process_single_file(self, fpath):

        data_list = []
        with open(fpath, 'r') as f:
            reader = csv.reader(f)

            next(reader)

            for row in reader:
                if len(row) == 3: data_list.append(row)
        
        data = np.array(data_list, dtype=np.float64)

        if self.crop:
            f_min, f_max = self.crop[0] * 1e9, self.crop[1] * 1e9
            mask = (data[:,0] >= f_min) & (data[:,0] <= f_max)
            data = data[mask]

        real = data[:, 1]
        imag = data[:, 2]

        r_norm = (real - real.mean()) / (real.std() + 1e-8)
        i_norm = (imag - imag.mean()) / (imag.std() + 1e-8)
        x_complex = np.stack([r_norm, i_norm], axis=0)
        
        mag = np.sqrt(real**2 + imag**2)
        m_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
        x_power = m_norm[np.newaxis, :]

        return (x_complex, x_power)


if __name__ == "__main__":

    ROOT = "/home/telstar/AIxthz/data/data"
    split_gen = Split_Config_Gen()

    config = split_gen.get_cross_angle_config('mod2')

    train_ds = RFVibeS11Dataset(
        ROOT, 
        crop=None, 
        split_config=config['train'],  
        split_mode='cross_angle_split'
    )
    
    val_ds = RFVibeS11Dataset(
        ROOT, 
        crop=None, 
        split_config=config['val'],   
        split_mode='cross_angle_split'
    )
    print(len(train_ds), len(val_ds))
    print(train_ds[0][0].shape)