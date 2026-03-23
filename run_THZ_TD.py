from datetime import datetime
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split
import yaml

from dataset.TD_dataset import RF_Time_Dataset
from dataset.split_config import Split_Spec_Generator
from metrics.classification_metrics import MCC_Metrics


def get_dataloaders(data_dir, batch_size, split_mode_full):
    """
    Function to get the dataloaders for training and validation datasets.
    
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the dataloaders.
        split_mode_full (str): Mode of splitting the dataset ('cross_distance_split' or 'random_split').
    
    Returns:
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
    """
    # Initialize the dataset configuration based on the split mode
    split_generator = Split_Spec_Generator()
    split_mode = split_mode_full.split("_mod")[0] if "_mod" in split_mode_full else split_mode_full

    if split_mode == 'cross_distance_split':
        # Get the dataset configuration for distance-based split
        mod = split_mode_full.split("_")[-1]
        dataset_config = split_generator.get_cross_distance_split(mod)
        # Create datasets for training and validation
        train_dataset = RF_Time_Dataset(data_dir, dataset_config['train_dataset'], split_mode)
        val_dataset = RF_Time_Dataset(data_dir, dataset_config['val_dataset'], split_mode)
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
        class_mapping = train_dataset.class_mapping
    elif split_mode == 'cross_angle_split':
        # Get the dataset configuration for angle-based split
        mod = split_mode_full.split("_")[-1]
        dataset_config = split_generator.get_cross_angle_split(mod)
        # Create datasets for training and validation
        train_dataset = RF_Time_Dataset(data_dir, dataset_config['train_dataset'], split_mode)
        val_dataset = RF_Time_Dataset(data_dir, dataset_config['val_dataset'], split_mode)
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
        class_mapping = train_dataset.class_mapping
    elif split_mode == 'random_split':
        # Get the full dataset for random split
        dataset_config = None
        full_dataset = RF_Time_Dataset(data_dir, None, split_mode)
        # Get the random split settings
        generator, train_size, val_size = split_generator.get_random_split(len(full_dataset))
        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
        class_mapping = full_dataset.class_mapping
    else:
        raise ValueError("Invalid split mode. Choose either 'cross_distance_split' or 'random_split'.") 
    return train_loader, val_loader, class_mapping, dataset_config


def train(model, train_loader, val_loader, logger, num_epochs, criterion, optimizer, scheduler, val_freq, train_device, n_classes, model_dir):
    """Function to train the model.
    
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        logger (logging.Logger): Logger for logging training information.
        num_epochs (int): Number of epochs to train the model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        val_freq (int): Frequency of validation during training.
        train_device (torch.device): Device to train the model on.
        n_classes (int): Number of classes in the dataset.
        model_dir (str): Directory to save the best model.

    """
    model = model.to(train_device)
    best_accuracy = 0
    best_epoch = 0
    best_model = None
    best_val_eval_metrics = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        training_eval_metrics = MCC_Metrics(n_classes=n_classes)

        i = 0

        for data in train_loader:
            # if i < 500:
            inputs,labels = data
            inputs = inputs.to(train_device)
            labels = labels.to(train_device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(train_device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            training_eval_metrics.evaluate(outputs,labels)
            # i += 1
        epoch_loss = epoch_loss/len(train_loader.dataset)
        # epoch_loss = epoch_loss/ (i * train_loader.batch_size)
        logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.9f}, Accuracy: {training_eval_metrics.accuracy():.4f}")

        if (epoch+1) % val_freq == 0:
            test_loss, val_eval_metrics = test(
                model=model,
                tensor_loader=val_loader,
                criterion=criterion,
                n_classes=n_classes,
                device= train_device
                )
            logger.info(f"Validation accuracy at epoch {epoch + 1}: {val_eval_metrics.accuracy():.4f}")

            if val_eval_metrics.accuracy() >= best_accuracy:
                best_accuracy = val_eval_metrics.accuracy()
                logger.info(f"New Best Accuracy: {best_accuracy:.4f}")
                best_model = model
                best_epoch = epoch
                best_val_eval_metrics = val_eval_metrics

        scheduler.step()
    # Save the best model
    model_save_path = os.path.join(model_dir, f"best_model_epoch_{best_epoch + 1}.pth")
    torch.save(best_model.state_dict(), model_save_path)
    logger.info(f"Best model saved at: {model_save_path}")
    logger.info(f"Best validation metrics at epoch {best_epoch + 1}: {best_val_eval_metrics}")
    return best_model


def test(model, tensor_loader, criterion, n_classes, device):
    model.eval()
    test_loss = 0
    val_eval_metrics = MCC_Metrics(n_classes=n_classes)
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        val_eval_metrics.evaluate(outputs,labels)
        test_loss += loss.item() * inputs.size(0)
    test_loss = test_loss/len(tensor_loader.dataset)
    
    return test_loss, val_eval_metrics


def run_time_main(config_path):
    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
       
    # Setup logging
    model_dir = os.path.join(config["save_dir"])
    log_path = os.path.join(model_dir, 'train.log')
    logger = logging.getLogger(config_path)  # unique logger name per config
    logger.setLevel(logging.INFO)

    # Remove old handlers to avoid duplicate logs
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    logger.info(f"Starting training with config: {config_path}")


    # get dataloaders and dataset configuration
    train_loader, val_loader, class_mapping, dataset_config = get_dataloaders(
        config["data_dir"],  
        config["batch_size"],
        config["split_mode"]
    )
    logger.info(f"Dataset configuration: {dataset_config}")
    n_classes = len(class_mapping)
    iter_train_loader = iter(train_loader)
    inputs, labels = next(iter_train_loader)
    logger.info(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class mapping: {class_mapping}")

    # Initialize the model based on the configuration
    if config["model"]["type"] == "LSTMResNetDualChannel":
        from model_zoo.lstm_resnet import LSTMResNetDualChannel
        model = LSTMResNetDualChannel(
            input_dim=config["model"]["params"]["input_dim"],
            hidden_dim=config["model"]["params"]["hidden_dim"],
            num_layers=config["model"]["params"]["num_layers"],
            num_classes=n_classes
        )
    elif config["model"]["type"] == "DeepComplexNet1D":
        from model_zoo.DCN import DeepComplexNet1D
        model = DeepComplexNet1D(
            input_length=config["model"]["params"]["input_length"],
            num_classes=n_classes
        )
    elif config["model"]["type"] == "Transformer1D":
        from model_zoo.transformer1d import Transformer1D
        model = Transformer1D(
            seq_len=config["model"]["params"]["seq_len"],
            in_dim=config["model"]["params"]["in_dim"],
            embed_dim=config["model"]["params"]["embed_dim"],
            num_heads=config["model"]["params"]["num_heads"],
            num_layers=config["model"]["params"]["num_layers"],
            num_classes=n_classes,
            # dropout=config["model"]["params"].get("dropout", 0.1)
        )
    elif config["model"]["type"] == "RF_MatID":
        from model_zoo.RF_MatID import RF_MatID
        model = RF_MatID(
            # seq_length=inputs.size(1),
            d_model=config["model"]["params"]["d_model"],
            num_classes=n_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")
    # print(model)
    # Load training configuration
    train_device = torch.device(config["training"]["train_device"] if torch.cuda.is_available() else "cpu")
    # Criterion
    criterion_cls = getattr(nn, config["training"]["criterion"]["type"])
    criterion = criterion_cls(**config["training"]["criterion"]["params"])
    # Optimizer
    optimizer_cls = getattr(torch.optim, config["training"]["optimizer"]["type"])
    optimizer = optimizer_cls(model.parameters(), **config["training"]["optimizer"]["params"])
    # Scheduler
    scheduler_cls = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"]["type"])
    scheduler = scheduler_cls(optimizer, **config["training"]["scheduler"]["params"])
    # Log training configuration
    logger.info(f"training on {train_device} in progressing ..................")
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
if __name__ == "__main__":
    run_time_main(config_path="C:\\Users\\Chen_Xinyan\\Desktop\\THZ\\Thz_x_AI\\config.yaml")
    print("Training completed. Check the log file for details.")