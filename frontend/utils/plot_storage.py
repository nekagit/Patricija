import os
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Union, Dict, Any
import json

class PlotStorage:
    """
    Utility class for managing plot storage with organized folder structure.
    """
    
    def __init__(self, base_path: str = "images"):
        """
        Initialize plot storage with base path.
        
        Args:
            base_path: Base directory for storing images
        """
        self.base_path = Path(base_path)
        self._ensure_base_structure()
    
    def _ensure_base_structure(self):
        """Ensure the base folder structure exists."""
        # Create main images directory
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "credit_checks").mkdir(exist_ok=True)
        (self.base_path / "analytics").mkdir(exist_ok=True)
        (self.base_path / "training").mkdir(exist_ok=True)
    
    def get_credit_check_path(self, application_id: str) -> Path:
        """
        Get the path for storing credit check plots for a specific application.
        
        Args:
            application_id: Unique identifier for the credit application
            
        Returns:
            Path object for the application's plot directory
        """
        app_path = self.base_path / "credit_checks" / application_id
        app_path.mkdir(exist_ok=True)
        return app_path
    
    def get_analytics_path(self, date: Optional[datetime] = None) -> Path:
        """
        Get the path for storing daily analytics plots.
        
        Args:
            date: Date for analytics (defaults to today)
            
        Returns:
            Path object for the analytics directory
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        analytics_path = self.base_path / "analytics" / date_str
        analytics_path.mkdir(exist_ok=True)
        return analytics_path
    
    def get_training_path(self, model_name: str, date: Optional[datetime] = None) -> Path:
        """
        Get the path for storing training-related plots.
        
        Args:
            model_name: Name of the model being trained
            date: Date for training (defaults to today)
            
        Returns:
            Path object for the training directory
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        training_path = self.base_path / "training" / date_str / model_name
        training_path.mkdir(parents=True, exist_ok=True)
        return training_path
    
    def save_matplotlib_plot(self, fig: plt.Figure, filename: str, 
                           application_id: Optional[str] = None,
                           analytics: bool = False,
                           training: bool = False,
                           model_name: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a matplotlib figure to the appropriate directory.
        
        Args:
            fig: Matplotlib figure object
            filename: Name of the file (without extension)
            application_id: Credit application ID (for credit check plots)
            analytics: Whether this is an analytics plot
            training: Whether this is a training plot
            model_name: Name of the model (for training plots)
            metadata: Additional metadata to save with the plot
            
        Returns:
            Full path to the saved image file
        """
        # Determine the appropriate directory
        if application_id:
            save_path = self.get_credit_check_path(application_id)
        elif analytics:
            save_path = self.get_analytics_path()
        elif training and model_name:
            save_path = self.get_training_path(model_name)
        else:
            save_path = self.get_analytics_path()
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{filename}_{timestamp}.png"
        file_path = save_path / unique_filename
        
        # Save the plot
        fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix('.json')
            metadata['filename'] = unique_filename
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['plot_type'] = 'matplotlib'
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            metadata = convert_numpy_types(metadata)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def save_plotly_plot(self, fig: Union[go.Figure, 'plotly.graph_objs._figure.Figure'], filename: str,
                        application_id: Optional[str] = None,
                        analytics: bool = False,
                        training: bool = False,
                        model_name: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        format: str = 'png') -> str:
        """
        Save a plotly figure to the appropriate directory.
        
        Args:
            fig: Plotly figure object
            filename: Name of the file (without extension)
            application_id: Credit application ID (for credit check plots)
            analytics: Whether this is an analytics plot
            training: Whether this is a training plot
            model_name: Name of the model (for training plots)
            metadata: Additional metadata to save with the plot
            format: Image format (png, jpg, svg, pdf)
            
        Returns:
            Full path to the saved image file
        """
        # Determine the appropriate directory
        if application_id:
            save_path = self.get_credit_check_path(application_id)
        elif analytics:
            save_path = self.get_analytics_path()
        elif training and model_name:
            save_path = self.get_training_path(model_name)
        else:
            save_path = self.get_analytics_path()
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{filename}_{timestamp}.{format}"
        file_path = save_path / unique_filename
        
        # Save the plot
        fig.write_image(str(file_path), width=1200, height=800, scale=2)
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.with_suffix('.json')
            metadata['filename'] = unique_filename
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['plot_type'] = 'plotly'
            metadata['format'] = format
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            metadata = convert_numpy_types(metadata)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def get_application_plots(self, application_id: str) -> Dict[str, str]:
        """
        Get all plots for a specific credit application.
        
        Args:
            application_id: Credit application ID
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        app_path = self.get_credit_check_path(application_id)
        plots = {}
        
        if app_path.exists():
            for file_path in app_path.glob("*.png"):
                if not file_path.name.endswith('_metadata.json'):
                    plots[file_path.stem] = str(file_path)
        
        return plots
    
    def get_daily_analytics_plots(self, date: Optional[datetime] = None) -> Dict[str, str]:
        """
        Get all analytics plots for a specific date.
        
        Args:
            date: Date to get plots for (defaults to today)
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        analytics_path = self.get_analytics_path(date)
        plots = {}
        
        if analytics_path.exists():
            for file_path in analytics_path.glob("*.png"):
                if not file_path.name.endswith('_metadata.json'):
                    plots[file_path.stem] = str(file_path)
        
        return plots
    
    def cleanup_old_plots(self, days_to_keep: int = 30):
        """
        Clean up old plot files to save disk space.
        
        Args:
            days_to_keep: Number of days to keep plots
        """
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat().st_mtime < cutoff_date:
                    file_path.unlink()
                    print(f"Deleted old plot: {file_path}")

# Global instance for easy access
plot_storage = PlotStorage()
