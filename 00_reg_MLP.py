import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging 
import traceback
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union, Callable, Any  
from pathlib import Path
from enum import Enum
from dataclasses import dataclass



logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [logging.FileHandler('learn_model.log'), logging.StreamHandler()]
                    )


class ScalerType(Enum):
    """Enumeration of available scaler types."""
    STANDARD = 'standard'
    MINMAX = 'minmax'
    MAXABS = 'maxabs'
    ROBUST = 'robust'
    NORMAL = 'normal'

class NoiseType(Enum):
    """Enumeration of available noise types."""
    NORMAL = 'normal'
    UNIFORM = 'uniform'
    POISSON = 'poisson'

class DatasetError(Exception):
    '''Base exception for dataset related errors.'''
    pass

class EmptyDatasetError(DatasetError):
    '''Raised when the loaded CSV file is empty.'''
    pass

class ColumnNotFoundError(DatasetError):
    '''Raised when a specified column is not found in the CSV.'''
    def __init__(self, column_name: str) -> None:
        """Initialize with the missing column name."""
        super().__init__(f"Column '{column_name}' not found in CSV.")

class InvalidScalerTypeError(DatasetError):
    '''Raised when an invalid scaler type is provided.'''
    def __init__(self, scaler_type: str) -> None:
        """Initialize with the invalid scaler type."""
        super().__init__(f"Invalid scaler_type: {scaler_type}")

class InvalidNoiseTypeError(DatasetError):
    '''Raised when an invalid noise type is provided.'''
    def __init__(self, noise_type: str) -> None:
        """Initialize with the invalid noise type."""
        super().__init__(f"Invalid noise_type: {noise_type}")

class ScalerFactory:
    """Factory class for creating scikit-learn scalers."""
    @staticmethod
    def create_scaler(scaler_type: ScalerType) -> Union[StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer]:
        """
        Creates a scikit-learn scaler based on the provided ScalerType.

        Args:
            scaler_type (ScalerType): The type of scaler to create.

        Returns:
            Union[StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer]:
                An instance of the specified scaler.

        Raises:
            InvalidScalerTypeError: If an invalid scaler type is provided.
        """
        if scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        elif scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        elif scaler_type == ScalerType.MAXABS:
            return MaxAbsScaler()
        elif scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        elif scaler_type == ScalerType.NORMAL:
            return Normalizer()
        else:
            raise InvalidScalerTypeError(scaler_type)


class NoiseGenerator:
    @staticmethod
    def generate_noise(noise_type: NoiseType, x_shape: Tuple[int, ...], noise_std: float, noise_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generates noise of a specified type.

        Args:
            noise_type: The type of noise to generate (NORMAL, UNIFORM, POISSON).
            x_shape: The shape of the noise array.
            noise_std: The standard deviation of the noise.
            noise_params: Optional parameters for the noise generation.

        Returns:
            A numpy array containing the generated noise.

        Raises:
            InvalidNoiseTypeError: If an invalid noise type is provided.
            DatasetError: If an error occurs during noise generation.
        """
        try:
            if noise_type == NoiseType.NORMAL:
                return np.random.normal(0, noise_std, size=x_shape).astype('float32')
            elif noise_type == NoiseType.UNIFORM:
                noise_params = noise_params or {'low': -noise_std, 'high': noise_std}
                return np.random.uniform(noise_params['low'], noise_params['high'], size=x_shape).astype('float32')
            elif noise_type == NoiseType.POISSON:
                noise_params = noise_params or {'lam': noise_std}
                return np.random.poisson(noise_params['lam'], size=x_shape).astype('float32')
            else:
                raise InvalidNoiseTypeError(noise_type)
        except ValueError as e:
            logging.error(f"Error generating noise: {e}")
            raise DatasetError(f"Error generating noise: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during noise generation: {e}")
            raise DatasetError(f"An unexpected error occurred during noise generation: {e}") 

@dataclass
class DatasetConfig:
    """Configuration class for dataset loading and preprocessing."""
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


class IDataset(Dataset):
    """Dataset class for loading and preprocessing data from a CSV file."""
    def __init__(self, config: DatasetConfig) -> None:
        """
        Initializes the IDataset object.

        Args:
            config (DatasetConfig): Configuration object containing dataset parameters.
        """
        self.config = config
        self.root = config.root
        self.csv_file = config.csv_file
        self.scaler = config.scaler
        self.scaler_type = config.scaler_type
        self.noise_type = config.noise_type
        self.noise_std = config.noise_std
        self.noise_params = config.noise_params
        self.scaling_factor = config.scaling_factor
        self.data = self._load_data(config.chunksize)

        if self.data.empty:
            raise EmptyDatasetError()

        for col in config.xcol + config.ycol:
            if col not in self.data.columns:
                raise ColumnNotFoundError(col)

        self.x = self.data[config.xcol].values.astype('float32').reshape(-1, len(config.xcol))
        self.y = self.data[config.ycol].values.astype('float32').reshape(-1, len(config.ycol))

        self._preprocess_data()


    def _load_data(self, chunksize: int = None) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            chunksize (int, optional): Chunk size for loading data in chunks. Defaults to None.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            EmptyDatasetError: If the CSV file is empty.
            DatasetError: If there is an error parsing the CSV file or an unexpected error occurs.
        """
        filepath = Path(self.root) / self.csv_file
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found at {filepath}")
        try:
            if chunksize is None:
                return pd.read_csv(filepath)
            else:
                chunks = pd.concat(pd.read_csv(filepath, chunksize=chunksize))
                return chunks
        except pd.errors.EmptyDataError:
            raise EmptyDatasetError()
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV file: {e}")
            raise DatasetError(f"Error parsing CSV file: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading data: {e}")
            raise DatasetError(f"An unexpected error occurred while loading data: {e}")
        

    def _preprocess_data(self) -> None:
        """Applies scaling, noise, and finalizes the data."""
        self._apply_scaling()
        self._apply_noise()
        self._finalize_data()

    def _apply_scaling(self) -> None:
        """Applies scaling to the input data."""
        try:
            self.scaler = ScalerFactory.create_scaler(self.scaler_type)
            self.x = self.scaler.fit_transform(self.x)
        except InvalidScalerTypeError as e:
            raise e
        except Exception as e:
            logging.error(f"Error applying scaling: {e}")
            raise DatasetError(f"Error applying scaling: {e}")
        

    def _apply_noise(self) -> None:
        """Applies noise to the input data."""
        try:
            noise = NoiseGenerator.generate_noise(self.noise_type, self.x.shape, self.noise_std, self.noise_params)
            self.x += noise
        except InvalidNoiseTypeError as e:
            raise e
        except Exception as e:
            logging.error(f"Error applying noise: {e}")
            raise DatasetError(f"Error applying noise: {e}")


    def _finalize_data(self) -> None:
        """Finalizes the data by applying scaling factor and converting to tensors."""
        self.x = self.x * self.scaling_factor
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data and target at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Data and target tensors.
        """
        return self.x[idx], self.y[idx]

    
class DeviceManager:
    """
    Manages device allocation and data transfer for PyTorch operations.
    """
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Initializes the DeviceManager.

        Args:
            device: Optional device specification (str or torch.device). If None, the default device is used.
        """
        if device is None:
            self.device = self._get_default_device()
        else:
            self.device = torch.device(device)
        logging.info(f"Using device: {self.device}")

    def _get_default_device(self) -> torch.device:
        """
        Determines the default device to use based on availability.

        Returns:
            torch.device: The default device (CUDA, MPS, or CPU).
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def to_device(self, data: Any) -> Any:
        """
        Transfers data to the managed device.

        Args:
            data: The data to transfer.

        Returns:
            Any: The data on the managed device.
        """
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking = True)

    def get_device(self) -> torch.device:
        """
        Returns the managed device.

        Returns:
            torch.device: The managed device.
        """
        return self.device


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


class IModel(nn.Module):
    """
    A generic neural network model with configurable hidden layers, dropout, and L1 regularization.
    """
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dims: List[int] = [32, 64, 32, 64, 128],
                 dropout_rate: float = 0.10, l1_lambda: float = 0.0) -> None:
        """
        Initializes the IModel.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            hidden_dims (List[int], optional): A list of hidden layer dimensions. Defaults to [32, 64, 32, 64, 128].
            dropout_rate (float, optional): The dropout rate. Defaults to 0.10.
            l1_lambda (float, optional): The L1 regularization lambda. Defaults to 0.0.

        Raises:
            ValueError: If input dimensions or parameters are invalid.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.l1_lambda = l1_lambda

        if not isinstance(in_dim, int) or in_dim <= 0:
            logging.error("in_dim must be a positive integer.")
            raise ValueError("in_dim must be a positive integer.")
        if not isinstance(out_dim, int) or out_dim <= 0:
            logging.error("out_dim must be a positive integer.")
            raise ValueError("out_dim must be a positive integer.")
        if not isinstance(hidden_dims, list) or not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
            logging.error("hidden_dims must be a list of positive integers.")
            raise ValueError("hidden_dims must be a list of positive integers.")
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate > 1:
            logging.error("dropout_rate must be a float between 0 and 1.")
            raise ValueError("dropout_rate must be a float between 0 and 1.")
        if not isinstance(l1_lambda, (int, float)) or l1_lambda < 0:
            logging.error("l1_lambda must be a non-negative float.")
            raise ValueError("l1_lambda must be a non-negative float.")

        layers = nn.ModuleList()
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(self._create_layer_block(dims[i], dims[i + 1]))
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.layers = nn.Sequential(*layers)

    def _create_layer_block(self, in_features: int, out_features: int) -> nn.Sequential:
        """
        Creates a layer block with Linear, BatchNorm, ELU, and Dropout.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            nn.Sequential: The layer block.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ELU(),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            TypeError: If the input is not a torch.Tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input 'x' must be a torch.Tensor.")
        return self.layers(x)

    def lasso_reg(self) -> torch.Tensor:
        """
        Calculates the L1 regularization term.

        Returns:
            torch.Tensor: The L1 regularization term.
        """
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.abs(param).sum()
        return self.l1_lambda * l1_norm
    

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = 'chk_point.pt',
        save_function: Optional[Callable[[dict, str], None]] = None,
    ) -> None:

        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'chk_point.pt'
            save_function (Callable): Custom function to save the model state. If None, uses torch.save.
        """
        
        if not isinstance(patience, int) or patience <= 0:
            logging.error("patience must be a positive integer.")
            raise ValueError("patience must be a positive integer.")
        if not isinstance(verbose, bool):
            logging.error("verbose must be a boolean.")
            raise TypeError("verbose must be a boolean.")
        if not isinstance(delta, (int, float)) or delta < 0:
            logging.error("delta must be a non-negative float.")
            raise ValueError("delta must be a non-negative float.")
        if not isinstance(path, str):
            logging.error("path must be a string.")
            raise TypeError("path must be a string.")

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.save_function = save_function if save_function else self._default_save_function
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.logger = logging.getLogger(__name__)

    def __call__(self, valid_loss: float, model: torch.nn.Module, epoch: int = 0) -> None:

        """
        Checks if the validation loss has improved and updates the best score accordingly.

        Args:
            valid_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            epoch (int): The current epoch number.
        """
        
        if not isinstance(valid_loss, (int, float)):
            logging.error("valid_loss must be a float or integer.)
            raise TypeError("valid_loss must be a float or integer.")
        if not isinstance(model, torch.nn.Module):
            logging.error("model must be a torch.nn.Module."))
            raise TypeError("model must be a torch.nn.Module.")
        if not isinstance(epoch, int) or epoch < 0:
            logging.error("epoch must be a non-negative integer.")
            raise ValueError("epoch must be a non-negative integer.")

        if self.best_score is None:
            self.best_score = valid_loss
            self._update_best_score(model, valid_loss)
            if self.verbose:
                self.logger.info(f'EarlyStopping - Initial Epoch: {epoch}, Validation Loss: {valid_loss:.6f}, Saving Model')
        elif self._is_improvement(valid_loss):
            if self.verbose:
                self.logger.info(f'EarlyStopping - Epoch: {epoch}, Validation Loss Improved: {self.best_score:.6f} --> {valid_loss:.6f}, Saving Model')
            self._update_best_score(model, valid_loss)
            self.counter = 0
        else:
            self._increment_counter(epoch)

    def _is_improvement(self, valid_loss: float) -> bool:
        return valid_loss < self.best_score - self.delta

    def _update_best_score(self, model: torch.nn.Module, valid_loss: float) -> None:
        self.best_score = valid_loss
        self.save_function({'model': model, 'state_dict': model.state_dict()}, self.path)

    def _increment_counter(self, epoch: int) -> None:
        """
        Increments the counter and checks if early stopping should be triggered.

        Args:
            epoch (int): The current epoch number.
        """
        
        self.counter += 1
        self.logger.info(f'EarlyStopping - Epoch: {epoch}, Validation Loss Counter: {self.counter} of {self.patience}')
        if self.counter > self.patience:
            self.early_stop = True

    def _default_save_function(self, state: dict, path: str) -> None:

        """
        Saves the model state to the specified path using torch.save.

        Args:
            state (dict): The model state dictionary.
            path (str): The path to save the model state.
        """
        torch.save(state, path)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    try:
        data_dir: Path = Path('C:/Chem_Data')

        if not data_dir.exists():
            logging.error(f"Directory not found: {data_dir}")
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        

        config = DatasetConfig(
            root=str(data_dir),
            xcol=['x_0', 'x_1'],
            ycol=['y_0', 'y_1', 'y_2'],
            scaler_type=ScalerType.NORMAL,
            noise_type=NoiseType.UNIFORM,
            noise_std=0.07,
            scaling_factor=60.0,
        )

        dataset = IDataset(config)

        in_dim = dataset.x.shape[1]
        out_dim = dataset.y.shape[1]

        def my_dataloaders(dataset,
                           train_ratio=0.8,
                           valid_ratio=0.19,
                           batch_size=9, seed=11,
                           shuffle_valid=False,
                           shuffle_test=False,
                           return_test=True):

            """
            Creates data loaders for training, validation, and optionally testing.

            Args:
                dataset: The dataset to be split and loaded.
                train_ratio (float): The proportion of the dataset to be used for training.
                valid_ratio (float): The proportion of the dataset to be used for validation.
                batch_size (int): The batch size for the data loaders.
                seed (int): The random seed for reproducibility.
                shuffle_valid (bool): Whether to shuffle the validation data.
                shuffle_test (bool): Whether to shuffle the test data.
                return_test (bool): Whether to return the test data loader.

            Returns:
                tuple: A tuple containing the train, validation, and optionally test data loaders.
                       If return_test is False, the test data loader will be None.

            Raises:
                ValueError: If the dataset is empty, the ratios are invalid, or the batch size is invalid.
            """
            

            if not dataset:
                raise ValueError("Dataset cannot be empty.")

            if train_ratio + valid_ratio > 1:
                raise ValueError("train_ratio + valid_ratio must be less than or equal to 1.")

            if batch_size <= 0 or not isinstance(batch_size, int):
                raise ValueError("Batch size must be a positive integer.")

            torch.manual_seed(seed)
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            valid_size = int(valid_ratio * total_size)
            test_size = total_size - train_size - valid_size

            if return_test:
                train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)
            else:
                train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size + test_size])
                test_loader = None

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_valid)

            if return_test:
                return train_loader, valid_loader, test_loader
            else:
                return train_loader, valid_loader

        config = {
            'model': {
                'in_dim': in_dim,
                'out_dim': out_dim,
                'l1_lambda': 0.01,
            },
            'training': {
                'criterion': {
                    'delta': 0.1,
                },
                'optimizer': {
                    'lr': 1e-1,
                    'weight_decay': 1e-3,
                },
                'scheduler': {
                    'step_lr': {
                        'step_size': 30,
                        'gamma': 0.1,
                    },
                    'reduce_lr': {
                        'factor': 0.1,
                        'patience': 10,
                        'mode': 'min',
                        'min_lr': 1e-6,
                    },
                },
                'early_stopping': {
                    'patience': 13,
                    'delta': 0.0001,
                },
                'batch_size': 9,
            },
        }

        def create_model(config: dict) -> 'IModel':

            """
            Creates an IModel instance based on the provided configuration.

            Args:
                config (dict): A dictionary containing model configuration parameters.
                               Expected keys:
                                   'model': A dictionary containing model-specific parameters.
                                            Expected keys within 'model':
                                                'in_dim' (int): Input dimension of the model.
                                                'out_dim' (int): Output dimension of the model.
                                                'l1_lambda' (float): L1 regularization lambda value.

            Returns:
                IModel: An instance of the IModel class, initialized with the specified parameters.
            """
            
            return IModel(in_dim=config['model']['in_dim'], out_dim=config['model']['out_dim'],
                          l1_lambda=config['model']['l1_lambda'])

        def create_criterion(config: Dict[str, Any]) -> nn.HuberLoss:

            """
            Creates a Huber Loss criterion based on the provided configuration.

            Args:
                config (Dict[str, Any]): A dictionary containing the training configuration,
                                         specifically the 'criterion' sub-dictionary with the
                                         'delta' value.

            Returns:
                nn.HuberLoss: An instance of the Huber Loss criterion with the specified delta
                               and reduction set to 'sum'.
            """
            return nn.HuberLoss(reduction='sum', delta=config['training']['criterion']['delta'])

        def create_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:

            """
            Creates an Adam optimizer for the given model with the specified learning rate and weight decay.

            Args:
                model (torch.nn.Module): The model for which to create the optimizer.
                config (dict): A dictionary containing the training configuration, including optimizer parameters.

            Returns:
                optim.Optimizer: An Adam optimizer instance.
            """
            return optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'],
                              weight_decay=config['training']['optimizer']['weight_decay'])

        def create_schedulers(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Tuple[StepLR, ReduceLROnPlateau]:

            """
            Creates and returns a tuple of learning rate schedulers.

            Args:
                optimizer (optim.Optimizer): The optimizer to be used with the schedulers.
                config (Dict[str, Any]): A dictionary containing the configuration for the schedulers.

            Returns:
                Tuple[StepLR, ReduceLROnPlateau]: A tuple containing the StepLR and ReduceLROnPlateau schedulers.
            """
            step_lr = StepLR(optimizer,
                                step_size=config['training']['scheduler']['step_lr']['step_size'],
                                gamma=config['training']['scheduler']['step_lr']['gamma'])
            red_lr = ReduceLROnPlateau(optimizer,
                                        factor=config['training']['scheduler']['reduce_lr']['factor'],
                                        patience=config['training']['scheduler']['reduce_lr']['patience'],
                                        mode=config['training']['scheduler']['reduce_lr']['mode'],
                                        min_lr=config['training']['scheduler']['reduce_lr']['min_lr'])
            return step_lr, red_lr

        def create_early_stopping(config: dict) -> object:
            """
            Creates an EarlyStopping object based on the provided configuration.

            Args:
                config (dict): A dictionary containing the training configuration,
                               including early stopping parameters.

            Returns:
                object: An EarlyStopping object configured with the specified parameters.
            """
            return EarlyStopping(patience=config['training']['early_stopping']['patience'], verbose=True,
                                    delta=config['training']['early_stopping']['delta'])

        def create_dataloaders(dataset: object, config: dict) -> tuple:
            """
            Creates dataloaders for training, validation, and testing.

            Args:
                dataset: The dataset object to create dataloaders from.
                config: A dictionary containing configuration parameters, including 'training' and 'batch_size'.

            Returns:
                A tuple containing the training, validation, and test dataloaders.
            """
            return my_dataloaders(dataset, batch_size=config['training']['batch_size'], return_test=True)

        """
        Creates and initializes the model, criterion, optimizer, schedulers, early stopping, and data loaders.

        Args:
            config: A configuration object containing necessary parameters.
            dataset: The dataset to be used for training, validation, and testing.

        Raises:
            FileNotFoundError: If a file specified in the configuration is not found.
            ValueError: If there is an issue with the configuration values.
            Exception: For any other unexpected errors during the process.
        """
        model = create_model(config)
        criterion = create_criterion(config)
        optimizer = create_optimizer(model, config)
        step_lr, red_lr = create_schedulers(optimizer, config)
        early_stopping = create_early_stopping(config)
        train_loader, valid_loader, test_loader = create_dataloaders(dataset, config)
        
        logging.info("Data loaders created successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


    def calculate_metrics(y_true: list, y_pred: list) -> tuple:
        """Calculates MSE, MAE, and R-squared with error handling.
        Args:
            y_true: A list of true target values.
            y_pred: A list of predicted target values.

        Returns:
            A tuple containing MSE, MAE, and R-squared. Returns (NaN, NaN, NaN)
            if an error occurs during calculation.
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return mse, mae, r2
        except ValueError as e:
            logging.error(f"Error calculating metrics: {e}")
            return float('nan'), float('nan'), float('nan')  # Return NaN in case of error
        except Exception as e:
            logging.error(f"An unexpected error occurred during metric calculation: {e}")
            traceback.print_exc()
            return float('nan'), float('nan'), float('nan')

    def collect_predictions_and_labels(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                       device: torch.device) -> tuple[list[float], list[int]]:
        """Collects predictions and labels with error handling.
        Args:
            model (nn.Module): The neural network model used for prediction.
            data_loader (torch.utils.data.DataLoader): The data loader providing batches of input data and labels.
            device (torch.device): The device (CPU or GPU) on which the model and data reside.

        Returns:
            tuple[list[float], list[int]]: A tuple containing two lists:
                - The first list contains the model's predictions as flattened floats.
                - The second list contains the true labels as flattened integers.
                Returns empty lists in case of an error.
        """
        all_preds, all_labels = [], []
        
        try:
            with torch.no_grad():  # Disable gradient calculations during evaluation
                for x_batch, y_batch in data_loader:
                    x_batch, y_batch = device_manager.to_device((x_batch, y_batch))
                    y_pred = model(x_batch)
                    all_preds.extend(y_pred.detach().cpu().numpy().flatten())
                    all_labels.extend(y_batch.cpu().numpy().flatten())
            return all_preds, all_labels
        except RuntimeError as e:
            logging.error(f"Runtime error during prediction collection: {e}")
            return [], []
        except Exception as e:
            logging.error(f"An unexpected error occurred during prediction and label collection: {e}")
            return [], []

    def evaluate(model: nn.Module, criterion: object, data_loader: torch.utils.data.DataLoader, epoch: int, mode: str, early_stopping: object = None) -> dict:
        """Evaluates the model with error handling.
        Args:
            model (nn.Module): The neural network model to evaluate.
            criterion (object): The loss function used for evaluation.
            data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
            epoch (int): The current epoch number (can be None).
            mode (str): The mode of evaluation ('train', 'valid', or 'test').
            early_stopping (object, optional): Early stopping object for validation. Defaults to None.

        Returns:
            dict: A dictionary containing evaluation metrics:
                - 'loss' (float): Average loss.
                - 'predictions' (list): List of model predictions.
                - 'labels' (list): List of true labels.
                - 'mse' (float): Mean Squared Error.
                - 'mae' (float): Mean Absolute Error.
                - 'r2' (float): R-squared value.
        """
        model.eval()
        total_loss = 0

        all_preds, all_labels = collect_predictions_and_labels(model, data_loader, device)

        if not all_preds or not all_labels:  # Check if prediction collection failed
            logging.error(f"Error: Prediction collection failed during {mode} evaluation.")
            return {"loss": float('nan'), "predictions": [], "labels": [], "mse": float('nan'), "mae": float('nan'), "r2": float('nan')}

        try:
            with torch.no_grad(): #disable gradients during evaluation.
                for x_batch, y_batch in data_loader:
                    x_batch, y_batch = device_manager.to_device((x_batch, y_batch))
                    y_pred = model(x_batch)
                    total_loss += criterion(y_pred, y_batch).item()

            average_loss = total_loss / len(data_loader)
            mse, mae, r2 = calculate_metrics(all_labels, all_preds)

            logging.info(f'Epoch{epoch + 1 if epoch is not None else ""}, {mode.capitalize()} Loss {average_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')

            if mode == "valid" and early_stopping is not None:
                early_stopping(average_loss, model, epoch)
                if early_stopping.early_stop:
                    logging.info('Early Stopping is here...')

            return {"loss": average_loss, "predictions": all_preds, "labels": all_labels, "mse": mse, "mae": mae, "r2": r2}
        except RuntimeError as e:
            logging.error(f"Runtime error during {mode} evaluation: {e}")
            return {"loss": float('nan'), "predictions": [], "labels": [], "mse": float('nan'), "mae": float('nan'), "r2": float('nan')}
        except Exception as e:
            logging.error(f"An unexpected error occurred during {mode} evaluation: {e}")
            traceback.print_exc()
            return {"loss": float('nan'), "predictions": [], "labels": [], "mse": float('nan'), "mae": float('nan'), "r2": float('nan')}

    device_manager = DeviceManager()
    device = device_manager.get_device()


    def train(model: nn.Module, criterion: Callable, optimizer: optim.Optimizer, step_lr: Callable, train_loader: torch.utils.data.DataLoader, epoch: int) -> dict:
        """Trains the model with error handling.
        Args:
            model (nn.Module): The neural network model to train.
            criterion (Callable): The loss function used for training.
            optimizer (optim.Optimizer): The optimizer used for updating model parameters.
            step_lr (Callable): The learning rate scheduler.
            train_loader (torch.utils.data.DataLoader): The data loader for the training set.
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing training metrics, including:
                - "loss" (float): The average training loss.
                - "predictions" (list): A list of model predictions.
                - "labels" (list): A list of true labels.
                - "mse" (float): Mean Squared Error.
                - "mae" (float): Mean Absolute Error.
                - "r2" (float): R-squared value.
                Returns NaN for all metrics in case of an error.
        """
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        try:
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                x_batch, y_batch = device_manager.to_device((x_batch, y_batch))
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                l1_reg = model.lasso_reg()
                loss += l1_reg
                total_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step_lr.step()
                all_preds.extend(y_pred.detach().cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

            average_loss = total_loss / len(train_loader)
            mse, mae, r2 = calculate_metrics(all_labels, all_preds)

            logging.info(f'Epoch{epoch + 1}, Training Loss {average_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}')

            return {"loss": average_loss, "predictions": all_preds, "labels": all_labels, "mse": mse, "mae": mae, "r2": r2}
        except RuntimeError as e:
            logging.error(f"Runtime error during training: {e}")
            return {"loss": float('nan'), "predictions": [], "labels": [], "mse": float('nan'), "mae": float('nan'), "r2": float('nan')}
        except Exception as e:
            logging.error(f"An unexpected error occurred during training: {e}")
            traceback.print_exc()
            return {"loss": float('nan'), "predictions": [], "labels": [], "mse": float('nan'), "mae": float('nan'), "r2": float('nan')}

    def run_training(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, step_lr: Callable,
                     train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, epochs: int,
                     early_stopping: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Runs the training loop with error handling.

        Args:
            model (nn.Module): The neural network model to train.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimization algorithm.
            step_lr (Callable): Learning rate scheduler.
            train_loader (DataLoader): DataLoader for the training dataset.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            epochs (int): The number of training epochs.
            early_stopping (Optional[Any], optional): Early stopping object. Defaults to None.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing training, validation, and test results.
                                       Returns an empty dictionary if an error occurs.
        """
        results = {}
        try:
            for epoch in range(epochs):
                train_results = train(model, criterion, optimizer, step_lr, train_loader, epoch,)
                valid_results = evaluate(model, criterion, valid_loader, epoch, "valid", early_stopping,)

                results[f"epoch_{epoch+1}"] = {"train": train_results, "valid": valid_results}
                if early_stopping is not None and early_stopping.early_stop:
                    break

            test_results = evaluate(model, criterion, test_loader, None, "test",)
            results["test"] = test_results

            return results
        except Exception as e:
            logging.error(f"An unexpected error occurred during the training loop: {e}")
            traceback.print_exc()
            return {} #return empty dict in case of a fatal error.
    

train_losses = []
valid_losses = []
test_losses = []
mse_values = []
mae_values = []
r2_values = []
epochs = []
test_preds = []
test_labels = []

try:
    results = run_training(model, criterion, optimizer, step_lr, train_loader, valid_loader, test_loader, epochs=1000, early_stopping=early_stopping,)

    # Accessing Results.
    if "test" not in results:
        raise ValueError("Test results not found in training output.")

    if "loss" not in results["test"] or "predictions" not in results["test"] or "labels" not in results["test"]:
        logging.error("Incomplete test results found.")
        raise ValueError("Incomplete test results found.")
        

    test_loss = results["test"]["loss"]
    test_preds = results["test"]["predictions"]
    test_labels = results["test"]["labels"]

    test_losses = [test_loss]  # make test_losses a list for consistency with other loss lists

    epoch_keys = [key for key in results if "epoch" in key]
    if not epoch_keys:
        raise ValueError("No epoch results found in training output.")

    train_losses = [results[key]["train"]["loss"] for key in epoch_keys if "train" in results[key] and "loss" in results[key]["train"]]
    valid_losses = [results[key]["valid"]["loss"] for key in epoch_keys if "valid" in results[key] and "loss" in results[key]["valid"]]
    mse_values = [results[key]["valid"]["mse"] for key in epoch_keys if "valid" in results[key] and "mse" in results[key]["valid"]]
    mae_values = [results[key]["valid"]["mae"] for key in epoch_keys if "valid" in results[key] and "mae" in results[key]["valid"]]
    r2_values = [results[key]["valid"]["r2"] for key in epoch_keys if "valid" in results[key] and "r2" in results[key]["valid"]]
    epochs = list(range(1, len(train_losses) + 1))

    if not train_losses or not valid_losses or not epochs:
        logging.error("Insufficient data to plot epoch-based metrics.")
        raise ValueError("Insufficient data to plot epoch-based metrics.")

    def plot_metrics(epochs: list, metrics: list[list], titles: list[str], y_labels: list[str], figure_size: tuple[int, int] = (12, 8)) -> None:
        """Plots multiple metrics in a grid.
        Args:
            epochs: A list of epoch numbers.
            metrics: A list of lists, where each inner list represents the metric values for each epoch.
            titles: A list of titles for each metric plot.
            y_labels: A list of y-axis labels for each metric plot.
            figure_size: The size of the figure (width, height) in inches. Defaults to (12, 8).

        Returns:
            None (displays the plot).
        """
        plt.figure(figsize=figure_size)
        for i, (metric, title, y_label) in enumerate(zip(metrics, titles, y_labels)):
            plt.subplot(2, 2, i + 1)
            plt.plot(epochs, metric, label=title)
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(y_label)
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals(test_labels: list, test_preds: list, figure_size: tuple = (8, 6)) -> None:
        """Plots residuals against predicted values.
        Args:
            test_labels (list): List of actual test labels.
            test_preds (list): List of predicted test labels.
            figure_size (tuple, optional): Size of the figure. Defaults to (8, 6).
        """
        residuals = np.array(test_labels) - np.array(test_preds)
        plt.figure(figsize=figure_size)
        plt.scatter(test_preds, residuals)
        plt.title('Residuals vs. Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

    metrics = [
        [train_losses, valid_losses],  # Training and Validation Loss as a nested list
        mse_values,
        mae_values,
        r2_values,
    ]
    titles = [
        'Training and Validation Loss',
        'Mean Squared Error (MSE)',
        'Mean Absolute Error (MAE)',
        'R-squared',
    ]
    y_labels = ['Loss', 'MSE', 'MAE', 'R-squared']

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics[0][0], label="Training Loss")
    plt.plot(epochs, metrics[0][1], label="Validation Loss")
    plt.title(titles[0])
    plt.xlabel("Epoch")
    plt.ylabel(y_labels[0])
    plt.legend()

    for i, (metric, title, y_label) in enumerate(zip(metrics[1:], titles[1:], y_labels[1:])):
        plt.subplot(2, 2, i + 2)
        plt.plot(epochs, metric, label=title)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend()
    plt.tight_layout()
    plt.show()

    plot_residuals(test_labels, test_preds)

except Exception as e:
    logging.error(f"An error occurred: {e}")


