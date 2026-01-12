"""
Configuration loader for YAML configuration files.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that configuration contains required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ['dataset', 'model', 'training', 'export']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset section
    dataset_required = ['equation', 'data_dir']
    for field in dataset_required:
        if field not in config['dataset']:
            raise ValueError(f"Missing required field in dataset section: {field}")
    
    # Validate model section
    if 'hidden_dim' not in config['model']:
        raise ValueError("Missing required field in model section: hidden_dim")
    
    if 'phase' not in config['model']:
        raise ValueError("Missing required field in model section: phase")
    
    # Validate training section
    if 'epochs' not in config['training']:
        raise ValueError("Missing required field in training section: epochs")
    
    if 'learning_rate' not in config['training']:
        raise ValueError("Missing required field in training section: learning_rate")

    # Optional: validate optimizer choice if provided
    optimizer = str(config["training"].get("optimizer", "adam")).lower()
    allowed_optimizers = {"adam", "adamw", "sgd"}
    if optimizer not in allowed_optimizers:
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. Allowed: {sorted(allowed_optimizers)}"
        )

    # Optional: validate weight_decay if provided
    if "weight_decay" in config["training"]:
        try:
            float(config["training"]["weight_decay"])
        except Exception as e:
            raise ValueError("training.weight_decay must be a float") from e
    
    # Validate export section
    if 'output_dir' not in config['export']:
        raise ValueError("Missing required field in export section: output_dir")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration template.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        'dataset': {
            'equation': 'convection',
            'data_dir': './data_gen/dataset',
            'seed': 42,
            'n_train': 10,
            'cache': True
        },
        'model': {
            'hidden_dim': 50,
            'phase': 'phase1'  # or 'phase2'
        },
        'training': {
            'epochs': 10000,
            'learning_rate': 0.00025,
            'batch_size': None,  # None means full batch
            'optimizer': 'adamw',
            'weight_decay': 0.0,
            'checkpoint_interval': 1000
        },
        'export': {
            'output_dir': './outputs',
            'save_plots': True,
            'plot_max_samples': 5
        },
        'device': 'cuda:0',
        'seed': 42
    }

