from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta

from models import (
    CreditApplication, Prediction, ModelTrainingHistory,
    ModelPerformanceLog, FeatureImportanceHistory, ApplicationAuditLog,
    SystemConfig, PredictionCache
)
from schemas import (
    CreditApplicationCreate, CreditApplicationUpdate,
    PredictionCreate, ModelTrainingCreate, ModelPerformanceCreate,
    FeatureImportanceCreate, AuditLogCreate, SystemConfigCreate, SystemConfigUpdate,
    PredictionCacheCreate
)

# Credit Application CRUD operations
def create_credit_application(db: Session, application: CreditApplicationCreate) -> CreditApplication:
    """Create a new credit application."""
    # Generate application number
    application_number = f"APP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    db_application = CreditApplication(
        application_number=application_number,
        **application.dict()
    )
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application

def get_credit_application(db: Session, application_id: uuid.UUID) -> Optional[CreditApplication]:
    """Get credit application by ID."""
    return db.query(CreditApplication).filter(CreditApplication.id == application_id).first()

def get_credit_applications(
    db: Session, 
    status: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[CreditApplication]:
    """Get credit applications with filters."""
    query = db.query(CreditApplication)
    
    if status:
        query = query.filter(CreditApplication.status == status)
    
    return query.offset(skip).limit(limit).all()

def update_credit_application(
    db: Session, 
    application_id: uuid.UUID, 
    application_update: CreditApplicationUpdate
) -> Optional[CreditApplication]:
    """Update credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        return None
    
    update_data = application_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_application, field, value)
    
    db.commit()
    db.refresh(db_application)
    return db_application

def delete_credit_application(db: Session, application_id: uuid.UUID) -> bool:
    """Delete credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        return False
    
    db.delete(db_application)
    db.commit()
    return True

# Prediction CRUD operations
def create_prediction(db: Session, prediction: PredictionCreate) -> Prediction:
    """Create a new prediction."""
    db_prediction = Prediction(**prediction.dict())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction(db: Session, prediction_id: uuid.UUID) -> Optional[Prediction]:
    """Get prediction by ID."""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions_by_application(
    db: Session, 
    application_id: uuid.UUID,
    skip: int = 0, 
    limit: int = 100
) -> List[Prediction]:
    """Get predictions for an application."""
    return db.query(Prediction).filter(
        Prediction.application_id == application_id
    ).offset(skip).limit(limit).all()

def get_predictions_by_model(
    db: Session, 
    model_name: str,
    skip: int = 0, 
    limit: int = 100
) -> List[Prediction]:
    """Get predictions by model name."""
    return db.query(Prediction).filter(
        Prediction.model_name == model_name
    ).offset(skip).limit(limit).all()

# Model Training CRUD operations
def create_model_training(db: Session, training: ModelTrainingCreate) -> ModelTrainingHistory:
    """Create a new model training record."""
    db_training = ModelTrainingHistory(**training.dict())
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    return db_training

def get_model_training(db: Session, training_id: uuid.UUID) -> Optional[ModelTrainingHistory]:
    """Get model training by ID."""
    return db.query(ModelTrainingHistory).filter(ModelTrainingHistory.id == training_id).first()

def get_model_training_history(
    db: Session, 
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100
) -> List[ModelTrainingHistory]:
    """Get model training history with filters."""
    query = db.query(ModelTrainingHistory)
    
    if model_name:
        query = query.filter(ModelTrainingHistory.model_name == model_name)
    if status:
        query = query.filter(ModelTrainingHistory.status == status)
    
    return query.order_by(desc(ModelTrainingHistory.training_completed_at)).offset(skip).limit(limit).all()

def update_model_training_status(
    db: Session, 
    training_id: uuid.UUID, 
    status: str,
    error_message: Optional[str] = None
) -> Optional[ModelTrainingHistory]:
    """Update model training status."""
    db_training = get_model_training(db, training_id)
    if not db_training:
        return None
    
    db_training.status = status
    if error_message:
        db_training.error_message = error_message
    
    db.commit()
    db.refresh(db_training)
    return db_training

# Model Performance CRUD operations
def create_model_performance(db: Session, performance: ModelPerformanceCreate) -> ModelPerformanceLog:
    """Create a new model performance log."""
    db_performance = ModelPerformanceLog(**performance.dict())
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    return db_performance

def get_model_performance_logs(
    db: Session, 
    model_name: Optional[str] = None,
    hours: int = 24,
    skip: int = 0, 
    limit: int = 100
) -> List[ModelPerformanceLog]:
    """Get model performance logs with filters."""
    query = db.query(ModelPerformanceLog)
    
    if model_name:
        query = query.filter(ModelPerformanceLog.model_name == model_name)
    
    # Filter by time range
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(ModelPerformanceLog.timestamp >= cutoff_time)
    
    return query.order_by(desc(ModelPerformanceLog.timestamp)).offset(skip).limit(limit).all()

# Feature Importance CRUD operations
def create_feature_importance(db: Session, feature_importance: FeatureImportanceCreate) -> FeatureImportanceHistory:
    """Create a new feature importance record."""
    db_feature_importance = FeatureImportanceHistory(**feature_importance.dict())
    db.add(db_feature_importance)
    db.commit()
    db.refresh(db_feature_importance)
    return db_feature_importance

def get_feature_importance_by_training(
    db: Session, 
    model_training_id: uuid.UUID
) -> List[FeatureImportanceHistory]:
    """Get feature importance for a model training."""
    return db.query(FeatureImportanceHistory).filter(
        FeatureImportanceHistory.model_training_id == model_training_id
    ).order_by(FeatureImportanceHistory.importance_rank).all()

# Audit Log CRUD operations (Simplified - no user authentication)
def create_audit_log(db: Session, audit_log: AuditLogCreate) -> ApplicationAuditLog:
    """Create a new audit log entry."""
    db_audit_log = ApplicationAuditLog(**audit_log.dict())
    db.add(db_audit_log)
    db.commit()
    db.refresh(db_audit_log)
    return db_audit_log

def get_audit_logs_by_application(
    db: Session, 
    application_id: uuid.UUID,
    skip: int = 0, 
    limit: int = 100
) -> List[ApplicationAuditLog]:
    """Get audit logs for an application."""
    return db.query(ApplicationAuditLog).filter(
        ApplicationAuditLog.application_id == application_id
    ).order_by(desc(ApplicationAuditLog.timestamp)).offset(skip).limit(limit).all()

# System Config CRUD operations
def create_system_config(db: Session, config: SystemConfigCreate) -> SystemConfig:
    """Create a new system configuration."""
    db_config = SystemConfig(**config.dict())
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config

def get_system_config(db: Session, config_key: str) -> Optional[SystemConfig]:
    """Get system configuration by key."""
    return db.query(SystemConfig).filter(
        and_(
            SystemConfig.config_key == config_key,
            SystemConfig.is_active == True
        )
    ).first()

def get_all_system_configs(db: Session, skip: int = 0, limit: int = 100) -> List[SystemConfig]:
    """Get all system configurations."""
    return db.query(SystemConfig).filter(
        SystemConfig.is_active == True
    ).offset(skip).limit(limit).all()

def update_system_config(
    db: Session, 
    config_key: str, 
    config_update: SystemConfigUpdate
) -> Optional[SystemConfig]:
    """Update system configuration."""
    db_config = get_system_config(db, config_key)
    if not db_config:
        return None
    
    update_data = config_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_config, field, value)
    
    db.commit()
    db.refresh(db_config)
    return db_config

# Prediction Cache CRUD operations
def create_prediction_cache(db: Session, cache_entry: PredictionCacheCreate) -> PredictionCache:
    """Create a new prediction cache entry."""
    db_cache = PredictionCache(**cache_entry.dict())
    db.add(db_cache)
    db.commit()
    db.refresh(db_cache)
    return db_cache

def get_prediction_cache(db: Session, input_hash: str) -> Optional[PredictionCache]:
    """Get prediction cache by input hash."""
    return db.query(PredictionCache).filter(
        and_(
            PredictionCache.input_hash == input_hash,
            PredictionCache.expires_at > datetime.utcnow()
        )
    ).first()

def update_cache_access(db: Session, cache_id: uuid.UUID):
    """Update cache access count and timestamp."""
    db_cache = db.query(PredictionCache).filter(PredictionCache.id == cache_id).first()
    if db_cache:
        db_cache.access_count += 1
        db_cache.last_accessed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_cache)

def cleanup_expired_cache(db: Session) -> int:
    """Clean up expired cache entries."""
    expired_count = db.query(PredictionCache).filter(
        PredictionCache.expires_at <= datetime.utcnow()
    ).count()
    
    db.query(PredictionCache).filter(
        PredictionCache.expires_at <= datetime.utcnow()
    ).delete()
    
    db.commit()
    return expired_count

# Analytics and reporting functions
def get_application_statistics(db: Session) -> Dict[str, Any]:
    """Get application statistics."""
    query = db.query(CreditApplication)
    
    total_applications = query.count()
    pending_applications = query.filter(CreditApplication.status == 'pending').count()
    approved_applications = query.filter(CreditApplication.status == 'approved').count()
    rejected_applications = query.filter(CreditApplication.status == 'rejected').count()
    
    return {
        "total": total_applications,
        "pending": pending_applications,
        "approved": approved_applications,
        "rejected": rejected_applications,
        "approval_rate": approved_applications / total_applications if total_applications > 0 else 0
    }

def get_prediction_statistics(db: Session, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get prediction statistics."""
    query = db.query(Prediction)
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    
    total_predictions = query.count()
    good_predictions = query.filter(Prediction.prediction == 'Good').count()
    bad_predictions = query.filter(Prediction.prediction == 'Bad').count()
    
    avg_processing_time = db.query(func.avg(Prediction.processing_time_ms)).scalar() or 0
    
    return {
        "total": total_predictions,
        "good": good_predictions,
        "bad": bad_predictions,
        "good_rate": good_predictions / total_predictions if total_predictions > 0 else 0,
        "avg_processing_time_ms": avg_processing_time
    }
