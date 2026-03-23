import torch
from torch.utils.data import DataLoader
import logging
import os
import sys
from datetime import datetime

from dataset import RFVibeS11Dataset, Split_Config_Gen
from model import RFVibeAdaptive 

# ==========================================
# 1. Logging config
# ==========================================
def setup_logger(save_dir="logs"):

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_file_dir, save_dir)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"exp_results_{timestamp}.log")

    logger = logging.getLogger("RFVibe_Exp")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    

    print(f"Saving path: {log_file}")

    return logger


# ==========================================
# 2. Training
# ==========================================
def run_training(train_ds, val_ds, num_classes, device, logger, epochs=60):
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    
    model = RFVibeAdaptive(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_c, x_p, y in train_loader:
            x_c, x_p, y = x_c.to(device), x_p.to(device), y.to(device)
            
            out_f, out_p, out_final = model(x_c, x_p)

            loss = 0.9 * criterion(out_f, y) + 0.3 * criterion(out_p, y) + 1.0 * criterion(out_final, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(out_final.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0.0
        
        log_message = f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%"


        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs: 
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for x_c, x_p, y in val_loader:
                    x_c, x_p, y = x_c.to(device), x_p.to(device), y.to(device)
                    _, _, out_final = model(x_c, x_p)
                    _, pred = torch.max(out_final, 1)
                    correct_val += (pred == y).sum().item()
                    total_val += y.size(0)
            
            val_acc = 100 * correct_val / total_val if total_val > 0 else 0.0
            if val_acc > best_acc: best_acc = val_acc
            
            log_message += f" | Val Acc: {val_acc:.2f}% | Best Val Acc: {best_acc:.2f}%"

        logger.info(log_message)
        
    return best_acc


if __name__ == "__main__":

    logger = setup_logger()
    
    ROOT = "/home/telstar/AIxthz/data/data/" 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SPLIT_GEN = Split_Config_Gen()
    
    logger.info(f"Device: {DEVICE}")
    logger.info(f"RootPath: {ROOT}")


    EXPERIMENTS = [
        # (任务名, crop, split_mode, mod_list)
        ("Full Band", (4.0, 43.5), ['random'], ['mod1', 'mod2', 'mod3']), 
        ("High Band", (30.0, 43.5), ['random'], ['mod1', 'mod2', 'mod3']),
        ("Low Band",  (4.0, 30.0),  ['random'], ['mod1', 'mod2', 'mod3']),
    ]
    
    results_log = []

    logger.info("="*60)
    logger.info("Start Experiments")
    logger.info("="*60)

    for band_name, crop_range, _, mods in EXPERIMENTS:
        
        logger.info(f"\n=== Now Working on: {band_name} {crop_range} ===")

        logger.info(f"[{band_name}] Running Random Split 70/30...")
        try:
            full_ds = RFVibeS11Dataset(ROOT, crop=crop_range, split_mode='random_split')
            train_sz = int(0.7 * len(full_ds))
            val_sz = len(full_ds) - train_sz
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(42))
            
            acc = run_training(train_ds, val_ds, len(full_ds.class_mapping), DEVICE, logger)
            logger.info(f"    Result: {acc:.2f}%")
            results_log.append(f"{band_name:<10} | Random Split   | N/A  : {acc:.2f}%")
        except Exception as e:
            logger.error(f"    Error in Random Split: {e}")
            results_log.append(f"{band_name:<10} | Random Split   | N/A  : ERROR")

        for mod in mods:
            
            logger.info(f">>> [{band_name}] Running Cross Distance ({mod})...")
            try:
                cfg = SPLIT_GEN.get_cross_distance_config(mod)
                train_ds = RFVibeS11Dataset(ROOT, crop=crop_range, split_config=cfg['train'], split_mode='cross_distance_split')
                val_ds = RFVibeS11Dataset(ROOT, crop=crop_range, split_config=cfg['val'], split_mode='cross_distance_split')
                
                acc = run_training(train_ds, val_ds, len(train_ds.class_mapping), DEVICE, logger)
                logger.info(f"    Result: {acc:.2f}%")
                results_log.append(f"{band_name:<10} | Cross Distance | {mod} : {acc:.2f}%")
            except Exception as e:
                logger.error(f"    Error in Cross Dist {mod}: {e}")
                results_log.append(f"{band_name:<10} | Cross Distance | {mod} : ERROR")

        for mod in mods:
            
            logger.info(f">>> [{band_name}] Running Cross Angle ({mod})...")
            try:
                cfg = SPLIT_GEN.get_cross_angle_config(mod)
                train_ds = RFVibeS11Dataset(ROOT, crop=crop_range, split_config=cfg['train'], split_mode='cross_angle_split')
                val_ds = RFVibeS11Dataset(ROOT, crop=crop_range, split_config=cfg['val'], split_mode='cross_angle_split')
                
                acc = run_training(train_ds, val_ds, len(train_ds.class_mapping), DEVICE, logger)
                logger.info(f"    Result: {acc:.2f}%")
                results_log.append(f"{band_name:<10} | Cross Angle    | {mod} : {acc:.2f}%")
            except Exception as e:
                logger.error(f"    Error in Cross Angle {mod}: {e}")
                results_log.append(f"{band_name:<10} | Cross Angle    | {mod} : ERROR")

    logger.info("\n" + "="*60)
    logger.info("Total Results：")
    logger.info("="*60)
    for line in results_log:
        logger.info(line)
    logger.info("="*60)