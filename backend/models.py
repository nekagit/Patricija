from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, Enum, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database import Base
from enum import Enum as PyEnum

# Custom UUID type for SQLite compatibility
class GUID(String):
    def __init__(self):
        super().__init__(36)
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.uuid4())
            else:
                return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)

class UserRole(PyEnum):
    user = "user"
    analyst = "analyst"
    admin = "admin"

class User(Base):
    __tablename__ = "users"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Personal Information
    first_name = Column(String(50))
    last_name = Column(String(50))
    role = Column(Enum(UserRole), default=UserRole.user, nullable=False)
    
    # Account Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    credit_applications = relationship("CreditApplication", back_populates="user")
    audit_logs = relationship("ApplicationAuditLog", back_populates="user")
    def __init__(self):
        super().__init__(36)
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.uuid4())
            else:
                return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)

class CreditApplication(Base):
    __tablename__ = "credit_applications"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    application_number = Column(String(20), unique=True, nullable=False)
    status = Column(Enum('pending', 'approved', 'rejected', 'under_review', name='application_status'), default='pending')
    
    # Personal Information
    person_age = Column(Integer, nullable=False)
    person_income = Column(Float, nullable=False)
    person_home_ownership = Column(Enum('RENT', 'OWN', 'MORTGAGE', 'OTHER', name='home_ownership'), nullable=False)
    person_emp_length = Column(Float, nullable=False)
    cb_person_cred_hist_length = Column(Integer, nullable=False)
    cb_person_default_on_file = Column(Enum('Y', 'N', name='default_status'), nullable=False)
    
    # Loan Information
    loan_intent = Column(Enum('PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION', name='loan_intent'), nullable=False)
    loan_grade = Column(Enum('A', 'B', 'C', 'D', 'E', 'F', name='loan_grade'), nullable=False)
    loan_amnt = Column(Float, nullable=False)
    loan_int_rate = Column(Float, nullable=False)
    loan_percent_income = Column(Float, nullable=False)
    
    # Application Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    submitted_at = Column(DateTime(timezone=True))
    notes = Column(Text)
    
    # User relationship
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    reviewed_by = Column(GUID(), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="credit_applications")
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    predictions = relationship("Prediction", back_populates="application")
    audit_logs = relationship("ApplicationAuditLog", back_populates="application")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    application_id = Column(GUID(), ForeignKey("credit_applications.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Prediction Results
    prediction = Column(Enum('Good', 'Bad', name='prediction_result'), nullable=False)
    probability_good = Column(Float, nullable=False)
    probability_bad = Column(Float, nullable=False)
    confidence_score = Column(Float)
    risk_category = Column(Enum('low', 'medium', 'high', name='risk_category'), nullable=False)
    
    # Model Input and Explanations
    input_features = Column(JSON, nullable=False)
    feature_importance = Column(JSON)
    shap_values = Column(JSON)
    lime_explanation = Column(JSON)
    feature_contributions = Column(JSON)
    
    # Metadata
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    input_hash = Column(String(64), nullable=False)
    
    # Relationships
    application = relationship("CreditApplication", back_populates="predictions")

class ModelTrainingHistory(Base):
    __tablename__ = "model_training_history"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    training_config = Column(JSON, nullable=False)
    
    # Training Data Info
    training_data_size = Column(Integer, nullable=False)
    test_data_size = Column(Integer, nullable=False)
    features_used = Column(JSON, nullable=False)  # Array as JSON
    target_variable = Column(String(50), nullable=False)
    
    # Performance Metrics
    accuracy = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training Details
    training_duration_seconds = Column(Integer)
    training_started_at = Column(DateTime(timezone=True), nullable=False)
    training_completed_at = Column(DateTime(timezone=True), nullable=False)
    
    # Model Storage
    model_file_path = Column(String(255))
    scaler_file_path = Column(String(255))
    encoders_file_path = Column(String(255))
    feature_names_file_path = Column(String(255))
    
    # Status
    status = Column(Enum('training', 'completed', 'failed', 'deployed', 'archived', name='training_status'), default='training')
    error_message = Column(Text)
    
    # Relationships
    feature_importance_history = relationship("FeatureImportanceHistory", back_populates="model_training")

class ModelPerformanceLog(Base):
    __tablename__ = "model_performance_logs"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Performance Metrics
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    prediction_count = Column(Integer, default=0)
    accuracy = Column(Float)
    avg_prediction_time_ms = Column(Float)
    cache_hit_rate = Column(Float)
    error_count = Column(Integer, default=0)
    
    # System Metrics
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)
    disk_usage_percent = Column(Float)
    active_connections = Column(Integer)

class FeatureImportanceHistory(Base):
    __tablename__ = "feature_importance_history"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_training_id = Column(GUID(), ForeignKey("model_training_history.id"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    importance_rank = Column(Integer, nullable=False)
    feature_type = Column(Enum('numerical', 'categorical', name='feature_type'), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model_training = relationship("ModelTrainingHistory", back_populates="feature_importance_history")

class ApplicationAuditLog(Base):
    __tablename__ = "application_audit_log"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    application_id = Column(GUID(), ForeignKey("credit_applications.id"), nullable=False)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    action = Column(Enum('created', 'updated', 'status_changed', 'prediction_made', 'reviewed', name='audit_action'), nullable=False)
    old_values = Column(JSON)
    new_values = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    
    # Relationships
    application = relationship("CreditApplication", back_populates="audit_logs")
    user = relationship("User", back_populates="audit_logs")

class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(Text, nullable=False)
    config_type = Column(Enum('string', 'integer', 'boolean', 'json', name='config_type'), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    input_hash = Column(String(64), unique=True, nullable=False)
    prediction_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
