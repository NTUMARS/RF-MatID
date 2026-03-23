import logging
import os
import torch
import torch.nn as nn
import yaml
from utils import get_dataloaders, set_seed, get_model, train

def run_main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
       
    # 2. Setup Logging
    model_dir = os.path.join(config["save_dir"])
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, 'train.log')
    
    logger = logging.getLogger(config_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Prevent duplicate logs if run multiple times

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    logger.info(f"Starting training with config: {config_path}")
    
    set_seed(config.get("seed", 3407))

    # 3. Data Loading
    # Handles both standard and 'application_mode' (v1) variations
    train_loader, val_loader, class_mapping, dataset_config = get_dataloaders(
        data_dir=config["data_dir"], 
        freq_range=config["freq_range"], 
        batch_size=config["batch_size"], 
        freq_data_type=config["freq_data_type"], 
        split_mode_full=config["split_mode"],
        application_mode=config.get("application_mode", 'all_classes') 
    )
    
    n_classes = len(class_mapping)
    inputs, labels = next(iter(train_loader))
    
    logger.info(f"Dataset configuration: {dataset_config}")
    logger.info(f"frequency range: {config['freq_range']}")
    logger.info(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class mapping: {class_mapping}")

    # 4. Model Initialization
    # We use get_model from utils to keep this main script clean of "model_zoo" imports
    model = get_model(config, n_classes, inputs)
    
    # 5. Training Setup
    train_device = torch.device(config["training"]["train_device"] if torch.cuda.is_available() else "cpu")
    
    # Loss, Optimizer, and Scheduler using dynamic getattr
    criterion_cls = getattr(nn, config["training"]["criterion"]["type"])
    criterion = criterion_cls(**config["training"]["criterion"]["params"])
    
    optimizer_cls = getattr(torch.optim, config["training"]["optimizer"]["type"])
    optimizer = optimizer_cls(model.parameters(), **config["training"]["optimizer"]["params"])

    scheduler_cls = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"]["type"])
    scheduler = scheduler_cls(optimizer, **config["training"]["scheduler"]["params"])

    # 6. Run Training
    logger.info(f"Training on {train_device}..................")
    best_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        num_epochs=config["training"]["train_epoch"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        val_freq=config["training"]["val_freq"],
        train_device=train_device,
        n_classes=n_classes,
        model_dir=model_dir
        )
    return best_model

if __name__ == "__main__":
    # Change this path to whichever experiment you are currently running
    CONFIG_FILE = "./config_hub/freq_domain/super_classes/freq_protocol_1/split_s1/mod0/RF_MatID/config.yaml"
    run_main(CONFIG_FILE)
    print("Training completed. Check the log file for details.")