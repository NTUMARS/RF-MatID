import os
import torch
from torch.utils.data import random_split
import random
import numpy as np

from dataset.split_config import Split_Spec_Generator
from dataset.dataset import RFMatID_Freq_Dataset
from metrics.classification_metrics import MCC_Metrics


def get_dataloaders(data_dir, freq_range, batch_size, freq_data_type, split_mode_full, application_mode):
    """
    Function to get the dataloaders for training and validation datasets.
    
    Args:
        data_dir (str): Directory containing the dataset.
        freq_range (tuple): Frequency range for the dataset.
        batch_size (int): Batch size for the dataloaders.
        freq_data_type (str): Type of frequency data ('complex' or 'dual_channel').
        split_mode_full (str): Splitting methods the dataset and corresponding sub mode (e.g. 'cross_distance_split_mod1').
        application_mode (str): Mode of application ('all_classes' or 'super_classes').
    
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
        train_dataset = RFMatID_Freq_Dataset(data_dir, freq_range, dataset_config['train_dataset'], freq_data_type, split_mode, application_mode)
        val_dataset = RFMatID_Freq_Dataset(data_dir, freq_range, dataset_config['val_dataset'], freq_data_type, split_mode, application_mode)
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
        class_mapping = train_dataset.class_mapping
    elif split_mode == 'cross_angle_split':
        # Get the dataset configuration for angle-based split
        mod = split_mode_full.split("_")[-1]
        dataset_config = split_generator.get_cross_angle_split(mod)
        # Create datasets for training and validation
        train_dataset = RFMatID_Freq_Dataset(data_dir, freq_range, dataset_config['train_dataset'], freq_data_type, split_mode, application_mode)
        val_dataset = RFMatID_Freq_Dataset(data_dir, freq_range, dataset_config['val_dataset'], freq_data_type, split_mode, application_mode)
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
        class_mapping = train_dataset.class_mapping
    elif split_mode == 'random_split':
        # Get the full dataset for random split
        dataset_config = None
        full_dataset = RFMatID_Freq_Dataset(data_dir, freq_range, None, freq_data_type, split_mode, application_mode)
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


def get_model(config, n_classes, inputs):
    seq_length = inputs.size(1)
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
    elif config["model"]["type"] == "ResNet50":
        from model_zoo.ResNet50 import resnet50_1d
        model = resnet50_1d(
            num_classes=n_classes,
            input_channels=config["model"]["params"]["input_channels"]
        )

    elif config["model"]["type"] == "DINOv3ConvNeXt":
        from model_zoo.DINOv3 import DINOv3ConvNeXt
        model = DINOv3ConvNeXt(
            num_classes=n_classes,
            patch_size=4,
            seq_len=1348,
            pretrained=config["model"]["params"].get("pretrained", True),
            freeze_backbone=config["model"]["params"].get("freeze_backbone", True)
        )

    elif config["model"]["type"] == "MLP":
        from model_zoo.MLP import MLP
        model = MLP(
            num_classes=n_classes,
            input_dim=config["model"]["params"]["input_dim"],
            expansion=config["model"]["params"]["expansion"]
        )

    elif config["model"]["type"] == "BiLSTM":
        from model_zoo.BiLSTM import BiLSTM
        model = BiLSTM(
            num_classes=n_classes,
            input_dim=config["model"]["params"]["input_dim"],
            hidden_size=config["model"]["params"]["hidden_size"],
            num_layers=config["model"]["params"]["num_layers"]
        )

    elif config["model"]["type"] == "ConvNeXt":
        from model_zoo.ConvNeXt.THZ_ConvNeXt import THZ_ConvNeXt
        model = THZ_ConvNeXt(
            num_classes=n_classes,
            input_dim=config["model"]["params"]["input_dim"],
            patch_size=config["model"]["params"]["patch_size"],
            seq_len=config["model"]["params"]["seq_len"]
        )

    elif config["model"]["type"] == "TimesNet":
        from model_zoo.TimesNet.TimesNet import TimesNet
        model = TimesNet(
            num_classes=n_classes,
            seq_len=config["model"]["params"]["seq_len"],
            input_dim=config["model"]["params"]["input_dim"]   
        )
    
    elif config["model"]["type"] == "MaterialID1D":
        from model_zoo.Material_ID import MaterialID1D
        model = MaterialID1D(
            in_channels=config["model"]["params"]["in_channels"],  # e.g. 2
            base_channels=config["model"]["params"].get("base_channels", 32),
            num_classes=n_classes
        )
    elif config["model"]["type"] == "MaterialID1DAdv":
        from model_zoo.Material_ID import MaterialID1DAdv
        model = MaterialID1DAdv(
            in_channels=config["model"]["params"]["in_channels"],
            base_channels=config["model"]["params"].get("base_channels", 32),
            num_classes=n_classes,
            use_distance_disc=config["model"]["params"].get("use_distance_disc", True),
            use_angle_disc=config["model"]["params"].get("use_angle_disc", True),
        )
    elif config["model"]["type"] == "AirTacMNet1D":
        from model_zoo.AirTac import AirTacMNet1D
        model = AirTacMNet1D(
            input_channels=config["model"]["params"]["input_channels"],
            seq_len=config["model"]["params"].get("seq_len", 2048),
            num_classes=n_classes,
            base_channels_mrf=config["model"]["params"].get("base_channels_mrf", 16),
            hidden_dim_mc=config["model"]["params"].get("hidden_dim_mc", 256),
        )
    elif config["model"]["type"] == "RF_MatID":
        from model_zoo.RF_MatID import RF_MatID
        model = RF_MatID(
            seq_length=seq_length,
            d_model=config["model"]["params"]["d_model"],
            drop_rate=config["model"]["params"].get("drop_rate", 0.0),
            num_classes=n_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")
    
    return model


def set_seed(seed: int = 3407):
    random.seed(seed)                # Python random
    np.random.seed(seed)             # NumPy random
    torch.manual_seed(seed)          # CPU
    torch.cuda.manual_seed(seed)     # Current GPU
    torch.cuda.manual_seed_all(seed)
    

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