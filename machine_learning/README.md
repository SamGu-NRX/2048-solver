# Machine Learning Directory

This directory contains the machine learning components for training and using AI models to play 2048.

## Files

- `BaseModel.hpp`: Abstract base class for all ML models
- `TD0.hpp`: Implementation of the TD0 (Temporal Difference Learning) algorithm
- `DoubleTD0.hpp`: Double TD0 model variant
- `ExportedTD0.hpp`: Wrapper for loading and using trained models
- `train.cpp`: Training script for TD0 models
- `train.sh`: Shell script to compile and run training
- `file_format.md`: Documentation of the binary model file format

## Model Directories

Each model directory contains trained weights for a specific configuration:

- `model_8-6_16_0.000150/`: 8 tuples, 6-tile size, max tile 16, learning rate 0.00015 (current best model)
- `model_12-5_15_0.000150/`: 12 tuples, 5-tile size, max tile 15, learning rate 0.00015
- `model_4-6_15_0.010000/`: 4 tuples, 6-tile size, max tile 15, learning rate 0.01
- `absmodel_8-6_12_0.000150__relmodel_8-6_7_0.000150/`: Combined absolute/relative model

### Model File Naming

Model files follow the pattern: `{model_name}_{epoch}.dat`
- Only final trained models (highest epoch number) are kept in version control
- Intermediate checkpoints are generated during training but excluded from the repository

## Training

To train a new model:
1. Configure parameters in `train.cpp`
2. Run `./train.sh` from the machine_learning directory
3. Model checkpoints will be saved at regular intervals
4. Training output logs are excluded from version control to save space

## Usage

The best trained model (`model_8-6_16_0.000150_1000.dat`) is loaded by `ExportedTD0.hpp` when `TESTING` is defined.
