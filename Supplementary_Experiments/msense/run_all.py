import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
import random

# ==========================================
# Split_Spec_Generator
# ==========================================
class Split_Spec_Generator():
    def __init__(self):
        self.all_distances = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                             1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600,
                             1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
        self.all_angles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def get_cross_distance_split(self, mod):
        if mod == 'mod1':
            val_distances = [250, 400, 550, 700, 900, 1050, 1200, 1350, 1550, 1700, 1900]
            train_distances = [d for d in self.all_distances if d not in val_distances]
        elif mod == 'mod2':
            train_distances = [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 
                               1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
            val_distances = [d for d in self.all_distances if d not in train_distances]
        elif mod == 'mod3':
            train_distances = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                               1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
            val_distances = [d for d in self.all_distances if d not in train_distances]
        else: return None, None
        return train_distances, val_distances
    
    def get_cross_angle_split(self, mod):
        if mod == 'mod1':
            train_angles = [0, 1, 3, 4, 6, 7, 9, 10]
            val_angles = [2, 5, 8]
        elif mod == 'mod2':
            train_angles = [3, 4, 5, 6, 7, 8, 9, 10]
            val_angles = [0, 1, 2]
        elif mod == 'mod3':
            train_angles = [0, 1, 2, 3, 4, 5, 6, 7]
            val_angles = [8, 9, 10]
        else: return None, None
        return train_angles, val_angles

# ==========================================
# Core Feature Extractor
# ==========================================
class mSense_Extractor:
    def __init__(self, root_dir, crop_range=None):
        self.root_dir = root_dir
        self.crop_range = crop_range
        self.C_SPEED = 299792458
        self.N_IFFT = 16384 
        
    def load_and_process(self, csv_path, known_dist_mm):

        data_list = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            next(reader)

            for row in reader:
                if len(row) == 3:
                    data_list.append(row)

        if not data_list:
            raise ValueError("NO datalist")

        data = np.array(data_list, dtype=np.float64)
        
        if self.crop_range:
            f_min, f_max = self.crop_range
            mask = (data[:,0] >= f_min*1e9) & (data[:,0] <= f_max*1e9)
            data = data[mask]


        freq = data[:,0]
        real = data[:,1]
        imag = data[:,2]
        
        n_data = len(freq)
        f_step = (freq[1] - freq[0]) if n_data > 1 else 0

        
        time_span = (1/f_step) * 1e9
        time_per_bin = time_span / self.N_IFFT
        
        s11 = real + 1j * imag
        cir = np.abs(np.fft.ifft(s11, self.N_IFFT))
        
        t0_bins = int(1.5 / time_per_bin)
        t0_idx = np.argmax(cir[0:t0_bins])
        
        known_dist_m = known_dist_mm / 1000.0
        expected_tau_ns = (known_dist_m * 2 / self.C_SPEED) * 1e9
        expected_offset_bins = int(expected_tau_ns / time_per_bin)
        
        expected_idx = t0_idx + expected_offset_bins



        window = 200
        f_window = 20
        search_start = max(t0_idx + 1, expected_idx - f_window)
        search_end = min(self.N_IFFT, expected_idx + window)


        target_region = cir[search_start:search_end]

        
        relative_peak_idx = np.argmax(target_region)
        final_target_idx = search_start + relative_peak_idx

        ad_raw = cir[final_target_idx]

        ad_t0 = cir[t0_idx]
        
        
        ad_normalized = ad_raw / ad_t0
        

        gamma = ad_normalized * known_dist_m

        return gamma
            
    def plot_cir(self, cir, final_target_idx, t0_idx, time_per_bin):
        N = len(cir)
        time_axis_ns = np.arange(N) * time_per_bin
        distance_axis_m = (time_axis_ns / 1e9) * self.C_SPEED / 2

        plt.figure(figsize=(14, 8))
        plt.plot(distance_axis_m, cir, label="CIR Amplitude")
        
        plt.axvline(distance_axis_m[t0_idx], color='g', linestyle=':', 
                    label=f't0 (Internal Reflection) at {distance_axis_m[t0_idx]:.3f}m (index {t0_idx})')
        
        plt.axvline(distance_axis_m[final_target_idx], color='r', linestyle='--', 
                    label=f'Target at {distance_axis_m[final_target_idx]:.3f}m (index {final_target_idx})')

        plt.title("TDR")
        plt.xlabel("Distance (m)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.xlim(-0.1, 2.0) 
        plt.legend()
        plt.show()


def simplify_label(raw_label):

    raw = raw_label.lower()
    if 'wood' in raw: return 'wood'
    if 'brick' in raw: return 'brick'
    if 'glass' in raw: return 'glass'
    if 'stone' in raw: return 'stone'
    if 'synthetic' in raw: return 'synthetic'
    return raw

if __name__ == "__main__":
    
    ROOT_DIR = "/home/telstar/AIxthz/data/data/"
    material_class = False

    FREQ_MODES = {
        'Full_Freq': None,
        'High_Freq (30-43.5)': (30.0, 43.5),
        'Low_Freq (4-30)': (4.0, 30.0)
    }
    
    SPLIT_GEN = Split_Spec_Generator()

    final_results = {}
    
    for freq_name, crop_range in FREQ_MODES.items():
        print(f"\n{'='*60}")
        print(f"Now in freq mod: {freq_name} ...")
        print(f"{'='*60}")
        
        # Create feature database for all samples
        extractor = mSense_Extractor(ROOT_DIR, crop_range)
        all_features_db = [] # {'label': 'wood', 'dist': 200, 'ang': 0, 'gamma': 0.123}
        
        all_folders = glob.glob(os.path.join(ROOT_DIR, '*'))
        print(f"Scanning {len(all_folders)} files...")
        
        for i, folder in enumerate(all_folders):
            if i % 500 == 0: print(f"Now at: {i}/{len(all_folders)}")
            
            try:
                dirname = os.path.basename(folder)
                parts = dirname.split('-')
                label = parts[-3]
                if material_class: 
                    label = simplify_label(label)
                dist = int(parts[-2].replace('mm',''))
                ang = int(parts[-1].replace('deg',''))
            except: raise ValueError("metadata false")
            
            csv_files = glob.glob(os.path.join(folder, '*.csv'))
            for fpath in csv_files:
                gamma = extractor.load_and_process(fpath, dist)
                if gamma is not None:
                    all_features_db.append({
                        'label': label,
                        'dist': dist,
                        'ang': ang,
                        'gamma': gamma
                    })
        
        db_len = len(all_features_db)
        print(f"Feature DB size: {db_len}")
        if db_len == 0: raise ValueError("NO DATABASE")

        tasks = []
        
        # a. (Random Split)
        tasks.append(('Random Split (70/30)', 'random', None))
        
        # b. Cross Distance
        for mod in ['mod1', 'mod2', 'mod3']:
            tasks.append((f'Cross Dist {mod}', 'distance', mod))
            
        # c. Cross Angle
        for mod in ['mod1', 'mod2', 'mod3']:
            tasks.append((f'Cross Angle {mod}', 'angle', mod))
            
        for task_name, mode, mod_arg in tasks:
            print(f"Testing: {task_name} ...", end="")
            
            train_set = []
            test_set = []
            
            if mode == 'random':
                shuffled_db = all_features_db.copy()
                random.seed(42)
                random.shuffle(shuffled_db)
                split_idx = int(len(shuffled_db) * 0.7)
                train_set = shuffled_db[:split_idx]
                test_set = shuffled_db[split_idx:]
                
            elif mode == 'distance':
                train_dists, val_dists = SPLIT_GEN.get_cross_distance_split(mod_arg)
                train_set = [x for x in all_features_db if x['dist'] in train_dists]
                test_set = [x for x in all_features_db if x['dist'] in val_dists]
                
            elif mode == 'angle':
                train_angs, val_angs = SPLIT_GEN.get_cross_angle_split(mod_arg)
                train_set = [x for x in all_features_db if x['ang'] in train_angs]
                test_set = [x for x in all_features_db if x['ang'] in val_angs]
            
            if not train_set or not test_set:
                print(" -> [continue: No enough data]")
                continue


            # fingerprint = {'wood01': 0.25, 'brick01': 0.8, ...}
            fingerprint_db = {}

            for item in train_set:
                lbl = item['label']
                val = item['gamma']
                if lbl not in fingerprint_db:
                    fingerprint_db[lbl] = []
                fingerprint_db[lbl].append(val)
            
            model_db = {}
            for mat, values in fingerprint_db.items():

                
                # "build a histogram... bin value and corresponding probability"
                counts, bin_edges = np.histogram(values, bins=30, density=True)
                
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                probs = counts / np.sum(counts)
                
                model_db[mat] = {
                    'gamma_i': bin_centers, # gamma_i(T) in paper
                    'p_i': probs            # p_i(T)
                }


            correct = 0
            for item in test_set:
                true_lbl = item['label']
                test_val = item['gamma']
                
                # L1 distance
                best_lbl = None
                min_score = float('inf')
                
                for mat_name, hist_data in model_db.items():
                    gamma_is = hist_data['gamma_i']
                    p_is = hist_data['p_i']
                    
                    distances = np.abs(test_val - gamma_is)
                    
                    score = np.sum(distances * p_is)
                    
                    if score < min_score:
                        min_score = score
                        best_lbl = mat_name
                
                if best_lbl == true_lbl:
                    correct += 1

            
            acc = (correct / len(test_set)) * 100
            print(f"Accuracy: {acc:.2f}%")
            

            key = f"{freq_name} | {task_name}"
            final_results[key] = acc

    # ==========================================
    # Final results
    # ==========================================
    print("\n\n" + "="*60)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<40} | {'Accuracy':<10}")
    print("-" * 55)
    
    for key, val in final_results.items():
        print(f"{key:<40} | {val:.2f}%")
    print("="*60)
