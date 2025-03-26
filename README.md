# PyTorch Regression Mutilayered Perceptron Model 

This repository contains a PyTorch implementation of a regression Mutilayered Perceptron model designed for tabular data. It includes data loading, preprocessing, model definition, training, and evaluation functionalities.

## Table of Contents 

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Dependencies](#dependencies)
- [Logging](#logging)
- [Error Handling](#error-handling)
- [Metrics and Visualization](#metrics-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Loading and Preprocessing:**
    - Loads data from CSV files.
    - Supports chunked loading for large datasets.
    - Handles missing columns and empty datasets.
    - Implements various data scaling techniques (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer).
    - Adds noise to the data (Normal, Uniform, Poisson).
    - Allows scaling and noise parameters to be configured.
- **Model Definition:**
    - Defines a customizable neural network model with configurable hidden layers, dropout, and L1 regularization.
    - Includes batch normalization and ELU activation.
- **Training and Evaluation:**
    - Implements training and validation loops with error handling.
    - Supports early stopping based on validation loss.
    - Uses Adam optimizer with configurable learning rate, weight decay, and learning rate schedulers (StepLR, ReduceLROnPlateau).
    - Calculates and logs metrics (MSE, MAE, R-squared).
    - Collects and visualizes training and validation losses, MSE, MAE, and R-squared.
    - Plots residuals against predicted values.
- **Device Management:**
    - Automatically detects and uses available GPU (CUDA or MPS) or CPU.
- **Logging:**
    - Uses Python's `logging` module to log training and evaluation information.
    - Logs to both a file (`learn_model.log`) and the console.
- **Error Handling:**
    - Includes custom exceptions for dataset-related errors.
    - Comprehensive error handling throughout the code.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required dependencies:

    ```bash
    pip install numpy pandas torch scikit-learn matplotlib
    ```

## Usage

1.  **Prepare your dataset:**
    - Place your CSV data file in the specified `root` directory.
    - Ensure the CSV file contains the columns specified in `xcol` and `ycol` within the `DatasetConfig`.

2.  **Configure the dataset:**
    - Modify the `DatasetConfig` in the `if __name__ == '__main__':` block to match your dataset.
    - Set the `root`, `csv_file`, `xcol`, `ycol`, `scaler_type`, `noise_type`, `noise_std`, and `scaling_factor` as needed.

3.  **Configure the model and training:**
    - Adjust the `config` dictionary in the `if __name__ == '__main__':` block to configure the model, optimizer, learning rate schedulers, and early stopping.

4.  **Run the script:**

    ```bash
    python <your_script_name>.py
    ```

5.  **View the results:**
    - The script will output training and evaluation metrics to the console and log file.
    - Plots of the training and validation losses, MSE, MAE, R-squared, and residuals will be displayed.

## Dataset Configuration

The `DatasetConfig` dataclass allows you to configure the dataset:

```python
@dataclass
class DatasetConfig:
    root: str
    xcol: list[str]
    ycol: list[str]
    scaler_type: ScalerType = ScalerType.STANDARD
    noise_type: NoiseType = NoiseType.NORMAL
    csv_file: str = 'dummy_data.csv'
    scaler: object = None
    noise_params: dict = None
    chunksize: int = None
    noise_std: float = 0.0
    scaling_factor: float = 1.0