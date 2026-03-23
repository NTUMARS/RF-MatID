# RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification

This repository is the official code implementation of the paper [RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](https://ntumars.github.io/project/RFMatID/) published on **ICLR 2026**.\
The paper presents the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification.
<!-- ![Motivation of RF-MatID](figures/teaser.png) -->

## 🤖Authors
- [Xinyan Chen](), [Qinchun Li](), [Ruiqin Ma](), [Jiaqi Bai](), [Li Yi](), [Jianfei Yang](https://marsyang.site/)
- [MARS Lab](http://marslab.tech/), Nanyang Technological University; Ibaraki University; SAP

## ⛷️Introduction
We introduce **RF-MatID**, a comprehensive dataset and benchmark designed to bridge the gap in robust material identification. While vision-based methods often struggle with optical ambiguity and lighting variations, RF-based sensing can reveal the intrinsic physical properties of materials.

**RF-MatID** is the largest and most diverse dataset of its kind, featuring:
- **16 fine-grained categories** derived from **5 superclasses** (e.g., Plastic, Metal, Wood, etc.).
- **Wide frequency range** from **4 GHz to 43.5 GHz**, capturing unique electromagnetic signatures across 39.5 GHz of bandwidth.
- **71k pairs of dual-domain samples**, providing both raw frequency-domain measurements and processed time-domain representations.
- **Geometry diversity**, with systematically controlled perturbations in incidence angles (0-10 deg)and stand-off distances (200-2000 mm).

RF-MatID also introduces a rigorous benchmark setting by defining **5 frequency protocols** (aligned with global regulations) and **7 data split settings** (including out-of-distribution tests). We extensively evaluate **11 deep learning models** (spanning CV, NLP, Time-Series, and RF research) across 3 protocols and 7 split settings. Additionally, a **comprehensive baseline** is constructed on all 5 protocols and 7 split settings.

## ⚙️Requirements

1. Install `pytorch` and `torchvision`.
2. `pip install -r requirements.txt`

## 🧾Prepare Datasets and PT Model Weights
### 🎄🪚Download Processed Data
- Please download the [RF-MatID Dataset](https://ntumars.github.io/project/RFMatID/) from the official project page.
- We recommend organizing the dataset in the following structure:
```
RF-MatID
├── RFMatID_Dataset
    ├── frequency_domain
    ├── time_domain
```


## 🏃‍♂️ Model Training & Evaluation
In the `config_hub` directory, we provide a wide range of **sample configuration files** covering 11 models (e.g., ResNet50, DINOv3, TimesNet) and experimental settings across 5 frequency protocols and 7 data splits mods.



### Dataset Loading & Perprocessing
In this section, I will introduce the **primary variables used for frequency protocols and dataset splitting**, which are essential for understanding how our benchmark are implemented and managed.

**`data_dir`**: Specifies the local file path to the root folder where the RF-MatID dataset is stored. E.g.
```
data_dir: "D:/Data/RFMatID_Dataset/frequency_domain"
```
**`freq_range`**: Defines the specific frequency bands to be used for training; setting it to `null` utilizes the full available bandwidth. E.g.
```
freq_range:
  - [4., 5.65]
  - [5.85, 10]
  - [10.7, 15.35]
  - [15.4, 23.6]
  - [24.25, 31.3]
  - [31.8, 36.43]
  - [36.5, 42.5]
```
**`batch_size`**: Determines the number of data samples processed in a single training step.
```
batch_size: 32
```
**`freq_data_type`**: Sets the input format for the RF signals, such as `complex`, `dual_channel`, or `three_channel` representations.
```
- complex: [real_part + j*imag_part, ...]
- dual_channel: [
    [real_part, imag_part], 
    [..., ...], ...]
- three_channel: [
    [frequency_position (in GHz), real_part, imag_part], 
    [..., ..., ...], ...]
```
**`split_mode`**: Controls the data partitioning strategy, allowing for standard random splits or Out-of-Distribution (OOD) tests like cross-distance and cross-angle scenarios.
```
- random_split
- cross_distance_split_mod1: 
    Interpolation using distance as the index.
- cross_distance_split_mod2: 
    Extrapolate using distance as the index; shorter distances for testing.
- cross_distance_split_mod3:
    Extrapolate using distance as the index; longer distances for testing.
- cross_angle_split_mod1: 
    Interpolation using angle as the index.
- cross_angle_split_mod2: 
    Extrapolate using angle as the index; smaller angles for testing.
- cross_distance_split_mod3:
    Extrapolate using angle as the index; larger angles for testing.
```
**`application_mode`**: Selects the classification granularity, ranging from all 16 fine-grained categories to the 5 broader super-classes.
```
- all_classes:
    Divide into 16 fine-grained categories.
- super_classes:
    Divide into 5 broader super-classes.
```

### Model Configuration
**`model.type`**: Identifies the specific neural network architecture to be instantiated.
```
- AirTacMNet1D
- BiLSTM
- ConvNeXt
- DINOv3ConvNeXt
- LSTMResNetDualChannel
- MaterialID1D
- MLP
- ResNet50
- TimesNet
- Transformer1D
- RF_MatID
```
**`model.params.XXX`**: 
Configures the detailed hyper-parameters associated with each model type. Refer to the examples provided in 
`config_hub/freq_domain/all_classes.freq_protocol_1/split_s1/mod0/*`

### Training & Optimization
**`training.train_device`**: Specifies whether the training should be executed on a GPU (`cuda`) or the `cpu`.

**`training.train_epoch`**: Defines the total number of times the model will iterate through the entire training dataset.

**`training.criterion.type`**: Selects the loss function used to measure the difference between predicted and actual labels.

**`training.optimizer.type`**: Chooses the optimization algorithm, such as AdamW, to update the model weights.

**`training.optimizer.params.lr`**: Sets the initial learning rate, which controls the step size of weight updates.

**`training.optimizer.params.weight_decay`**: Applies a L2 penalty to the weights to improve the model's generalization capability.

**`training.scheduler.type`**: Defines the strategy for adjusting the learning rate dynamically over the course of training.

**`training.scheduler.params.max_lr`**: Sets the peak learning rate reached during the scheduler's cycle (e.g., for OneCycleLR).

**`training.val_freq`**: Determines the interval (in epochs) at which the model is evaluated on the validation set.

### Output

**`save_dir`**: Specifies the directory where training logs, metrics, and the final model checkpoints will be saved.

## Run
```
python run.py --config [path/to/corresponding/config/file]
```

## ❤️‍🔥Citation
```bibtex
@article{chen2026rf,
        title={RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification},
        author={Chen, Xinyan and Li, Qinchun and Ma, Ruiqin and Bai, Jiaqi and Yi, Li and Yang, Jianfei},
        journal={arXiv preprint arXiv:2601.20377},
        year={2026}
}
```
