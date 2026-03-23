import torch


class Split_Spec_Generator():
    def __init__(self):
        r'''
        get the split settings for different split modes
            
        returns:
            dataset_config: dict, dictionary containing the split settings for training and validation datasets
        '''
        self.all_distances = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                        1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600,
                        1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
        self.all_angles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
    def get_cross_distance_split(self, mod):
        r'''
        Get the cross distance split settings for the dataset

        Args:
            mod: str, mode of splitting the dataset under cross distance split
            
        Returns:
            dataset_config: dict, dictionary containing the split settings for training and validation datasets
        '''
        train_form = {}
        val_form = {}
        if mod == 'mod1':
            val_distances = [250, 400, 550, 700, 900, 1050, 1200, 1350, 1550, 1700, 1900]
            train_distances = [200, 300, 350, 450, 500, 600, 650, 750, 800, 850, 950, 1000, 1100, 1150, 1250, 
                              1300, 1400, 1450, 1500, 1600, 1650, 1750, 1800, 1850, 1950, 2000]
        elif mod == 'mod2':
            train_distances = [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 
                               1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
            val_distances = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        elif mod == 'mod3':
            train_distances = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                            1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
            val_distances = [1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
        else:
            raise ValueError(f"mod is {mod}, mod must be either 'mod1', 'mod2' or 'mod3'")

        train_form['distances'] = train_distances
        val_form['distances'] = val_distances
        dataset_config = {'train_dataset': {
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                            'val_dataset': {
                                            'split': 'validation',
                                            'data_form': val_form}}
        return dataset_config
    
    def get_cross_angle_split(self, mod):
        r'''
        Get the cross angle split settings for the dataset
        
        Args:
            mod: str, mode of splitting the dataset under cross angle split

        Returns:
            dataset_config: dict, dictionary containing the split settings for training and validation datasets
        '''
        train_form = {}
        val_form = {}
        if mod == 'mod1':
            train_angles = [0, 1, 3, 4, 6, 7, 9, 10]
            val_angles = [2, 5, 8]
        elif mod == 'mod2':
            train_angles = [3, 4, 5, 6, 7, 8, 9, 10]
            val_angles = [0, 1, 2]
        elif mod == 'mod3':
            train_angles = [0, 1, 2, 3, 4, 5, 6, 7]
            val_angles = [8, 9, 10]

        train_form['angles'] = train_angles
        val_form['angles'] = val_angles
        dataset_config = {'train_dataset': {
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                            'val_dataset': {
                                            'split': 'validation',
                                            'data_form': val_form}}
        return dataset_config
        
    def get_random_split(self, full_dataset_size):
        r'''
        Get the random split settings for the dataset
        
        Args:
            full_dataset_size: int, size of the full dataset
            
        Returns:
            generator: torch.Generator, random number generator with a fixed seed
            train_size: int, size of the training dataset
            val_size: int, size of the validation dataset
        '''
        seed = 42
        generator = torch.Generator().manual_seed(seed)
        train_size = int(0.7 * full_dataset_size)
        val_size = full_dataset_size - train_size
        return generator, train_size, val_size