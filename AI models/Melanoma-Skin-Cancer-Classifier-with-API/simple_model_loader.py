"""
Simple model loader for scikit-learn models
"""
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_sklearn_model(filepath):
    """
    Load a scikit-learn model with basic error handling
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Loaded model object or None if loading fails
    """
    try:
        logger.info(f"Loading model from {filepath}")
        model = joblib.load(filepath)
        
        # Basic validation
        if hasattr(model, 'predict'):
            logger.info(f"Model loaded successfully from {filepath}")
            return model
        else:
            logger.error(f"Loaded object from {filepath} does not have predict method")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {str(e)}")
        return None

def find_and_load_model():
    """
    Find and load the first available model
    
    Returns:
        tuple: (model, model_name) or (None, "No model loaded")
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Priority order of models to try
    model_files = [
        ("melanoma-skin-cancer_vgg19.pkl", "VGG19 Melanoma Classifier"),
        ("melanoma.efficientnetb7.pkl", "EfficientNetB7 Melanoma Classifier"),
    ]
    
    for model_file, model_name in model_files:
        # Look for model file in the same directory as this script
        model_path = script_dir / model_file
        if model_path.exists():
            model = load_sklearn_model(model_path)
            if model is not None:
                return model, model_name
    
    return None, "No model loaded" 