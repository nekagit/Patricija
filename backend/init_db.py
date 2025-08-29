#!/usr/bin/env python3
"""
Database initialization script for Credit Check Application
Creates tables and inserts default system configuration
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import engine, Base
from models import (
    CreditApplication, Prediction, ModelTrainingHistory,
    ModelPerformanceLog, FeatureImportanceHistory, ApplicationAuditLog,
    SystemConfig, PredictionCache
)
from crud import create_system_config
from schemas import SystemConfigCreate
import uuid

def init_database():
    """Initialize the database with tables and default data."""
    print("Creating database tables...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")
    
    # Import database session
    from database import SessionLocal
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Insert default system configuration
        print("Inserting default system configuration...")
        
        default_configs = [
            {
                "config_key": "default_model_name",
                "config_value": "random_forest",
                "config_type": "string",
                "description": "Default model name for predictions"
            },
            {
                "config_key": "default_model_version",
                "config_value": "1.0.0",
                "config_type": "string",
                "description": "Default model version"
            },
            {
                "config_key": "cache_expiration_hours",
                "config_value": "24",
                "config_type": "integer",
                "description": "Prediction cache expiration time in hours"
            },
            {
                "config_key": "max_predictions_per_request",
                "config_value": "100",
                "config_type": "integer",
                "description": "Maximum number of predictions per API request"
            },
            {
                "config_key": "enable_prediction_cache",
                "config_value": "true",
                "config_type": "boolean",
                "description": "Enable prediction caching"
            },
            {
                "config_key": "log_level",
                "config_value": "INFO",
                "config_type": "string",
                "description": "Application log level"
            }
        ]
        
        for config_data in default_configs:
            # Check if config already exists
            existing_config = db.query(SystemConfig).filter(
                SystemConfig.config_key == config_data["config_key"]
            ).first()
            
            if not existing_config:
                config = SystemConfigCreate(**config_data)
                create_system_config(db, config)
                print(f"✓ Created config: {config_data['config_key']}")
            else:
                print(f"✓ Config already exists: {config_data['config_key']}")
        
        print("✓ Database initialization completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during database initialization: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Credit Check Application - Database Initialization")
    print("=" * 50)
    
    try:
        init_database()
        print("\nDatabase is ready for use!")
    except Exception as e:
        print(f"\nFailed to initialize database: {e}")
        sys.exit(1)
