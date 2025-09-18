# config_manager.py
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration dataclass for MultiClassifier"""
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    metrics: list = None
    epochs: int = 100
    batch_size: int = 32
    neurons: int = 32
    dropout: float = 0.5
    pred_threshold: float = 75.0
    validation_split: float = 0.2
    patience: int = 10

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy"]

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    user: str = "unknown"
    experiment_name: str = "default_experiment"

class ConfigManager:
    """Handles all configuration loading and management"""
    
    @staticmethod
    def find_config_file(custom_path: Optional[str] = None) -> Path:
        """Find configuration file in order of priority"""
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)
        
        config_path = os.environ.get("MLFLOW_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            return Path(config_path)
        
        # Try current working directory
        cwd_config = Path(os.getcwd()) / "config.json"
        if cwd_config.exists():
            return cwd_config
        
        # Try script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.json"
        if config_path.exists():
            return config_path
        
        # Legacy support
        legacy_config = Path(os.getcwd()) / "mlflow_config.json"
        if legacy_config.exists():
            return legacy_config
        legacy_config_alt = script_dir / "mlflow_config.json"
        if legacy_config_alt.exists():
            return legacy_config_alt
        
        raise FileNotFoundError(
            "Could not find config.json. Set MLFLOW_CONFIG_PATH or place config.json in working directory."
        )

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load raw configuration from JSON file"""
        path = ConfigManager.find_config_file(config_path)
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def create_model_config(raw_config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
        """Create ModelConfig from raw config with optional overrides"""
        config_dict = {}
        
        # Map JSON keys to ModelConfig fields
        key_mapping = {
            "OPTIMIZER": "optimizer",
            "LOSS": "loss", 
            "METRICS": "metrics",
            "EPOCHS": "epochs",
            "BATCH_SIZE": "batch_size",
            "NEURONS": "neurons",
            "DROPOUT": "dropout",
            "PRED_THRESHOLD": "pred_threshold",
            "VALIDATION_SPLIT": "validation_split",
            "PATIENCE": "patience"
        }
        
        for json_key, field_name in key_mapping.items():
            if json_key in raw_config:
                config_dict[field_name] = raw_config[json_key]
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if key in key_mapping.values():
                    config_dict[key] = value
                elif key.upper() in key_mapping:
                    config_dict[key_mapping[key.upper()]] = value
        
        return ModelConfig(**config_dict)

    @staticmethod
    def create_experiment_config(raw_config: Dict[str, Any]) -> ExperimentConfig:
        """Create ExperimentConfig from raw config"""
        return ExperimentConfig(
            user=raw_config.get("USER", "unknown"),
            experiment_name=raw_config.get("EXPERIMENT_NAME", "default_experiment")
        )

    @staticmethod
    def load_model_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
        """Load ModelConfig from file with optional overrides."""
        raw_config = ConfigManager.load_config(config_path)
        return ConfigManager.create_model_config(raw_config, overrides)