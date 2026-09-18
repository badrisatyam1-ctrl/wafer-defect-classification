"""Models module for wafer defect classification."""

try:
    from .wafer_cnn import (
        create_simple_cnn, 
        create_residual_cnn, 
        create_improved_cnn,
        FocalLoss,
        DEFECT_CLASSES
    )

    __all__ = [
        'create_simple_cnn', 
        'create_residual_cnn', 
        'create_improved_cnn',
        'FocalLoss',
        'DEFECT_CLASSES'
    ]
except ImportError:
    # TensorFlow not installed, skip loading TF models
    __all__ = []
