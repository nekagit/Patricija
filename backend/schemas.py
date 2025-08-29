from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

# Enums
class ApplicationStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    under_review = "under_review"

class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"

class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"

class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"

class DefaultStatus(str, Enum):
    Y = "Y"
    N = "N"

class PredictionResult(str, Enum):
    Good = "Good"
    Bad = "Bad"

class RiskCategory(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class TrainingStatus(str, Enum):
    training = "training"
    completed = "completed"
    failed = "failed"
    deployed = "deployed"
    archived = "archived"

class AuditAction(str, Enum):
    created = "created"
    updated = "updated"
    status_changed = "status_changed"
    prediction_made = "prediction_made"
    reviewed = "reviewed"

class ConfigType(str, Enum):
    string = "string"
    integer = "integer"
    boolean = "boolean"
    json = "json"

class FeatureType(str, Enum):
    numerical = "numerical"
    categorical = "categorical"

class UserRole(str, Enum):
    user = "user"
    analyst = "analyst"
    admin = "admin"

# Base schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# Credit Application schemas
class CreditApplicationBase(BaseSchema):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=0)
    person_home_ownership: HomeOwnership
    person_emp_length: float = Field(..., ge=0, le=50)
    cb_person_cred_hist_length: int = Field(..., ge=0, le=50)
    cb_person_default_on_file: DefaultStatus
    loan_intent: LoanIntent
    loan_grade: LoanGrade
    loan_amnt: float = Field(..., ge=100, le=1000000)
    loan_int_rate: float = Field(..., ge=0, le=100)
    loan_percent_income: float = Field(..., ge=0.0, le=1.0)

class CreditApplicationCreate(CreditApplicationBase):
    pass

class CreditApplicationUpdate(BaseSchema):
    person_age: Optional[int] = Field(None, ge=18, le=100)
    person_income: Optional[float] = Field(None, ge=0)
    person_home_ownership: Optional[HomeOwnership] = None
    person_emp_length: Optional[float] = Field(None, ge=0, le=50)
    cb_person_cred_hist_length: Optional[int] = Field(None, ge=0, le=50)
    cb_person_default_on_file: Optional[DefaultStatus] = None
    loan_intent: Optional[LoanIntent] = None
    loan_grade: Optional[LoanGrade] = None
    loan_amnt: Optional[float] = Field(None, ge=100, le=1000000)
    loan_int_rate: Optional[float] = Field(None, ge=0, le=100)
    loan_percent_income: Optional[float] = Field(None, ge=0.0, le=1.0)
    status: Optional[ApplicationStatus] = None
    notes: Optional[str] = None

class CreditApplicationResponse(CreditApplicationBase):
    id: uuid.UUID
    application_number: str
    status: ApplicationStatus
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None
    notes: Optional[str] = None

# Prediction schemas
class PredictionBase(BaseSchema):
    model_name: str = Field(..., max_length=100)
    model_version: str = Field(..., max_length=20)
    prediction: PredictionResult
    probability_good: float = Field(..., ge=0.0, le=1.0)
    probability_bad: float = Field(..., ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk_category: RiskCategory
    input_features: Dict[str, Any]
    feature_importance: Optional[Dict[str, Any]] = None
    shap_values: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    feature_contributions: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    cache_hit: bool = False
    input_hash: str = Field(..., max_length=64)

class PredictionCreate(PredictionBase):
    application_id: uuid.UUID

class PredictionResponse(PredictionBase):
    id: uuid.UUID
    application_id: uuid.UUID
    prediction_timestamp: datetime

# Model Training schemas
class ModelTrainingBase(BaseSchema):
    model_name: str = Field(..., max_length=100)
    model_version: str = Field(..., max_length=20)
    training_config: Dict[str, Any]
    training_data_size: int = Field(..., gt=0)
    test_data_size: int = Field(..., gt=0)
    features_used: List[str]
    target_variable: str = Field(..., max_length=50)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    pr_auc: float = Field(..., ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    training_duration_seconds: Optional[int] = None
    training_started_at: datetime
    training_completed_at: datetime
    model_file_path: Optional[str] = None
    scaler_file_path: Optional[str] = None
    encoders_file_path: Optional[str] = None
    feature_names_file_path: Optional[str] = None
    status: TrainingStatus = TrainingStatus.training
    error_message: Optional[str] = None

class ModelTrainingCreate(ModelTrainingBase):
    pass

class ModelTrainingResponse(ModelTrainingBase):
    id: uuid.UUID

# Model Performance schemas
class ModelPerformanceBase(BaseSchema):
    model_name: str = Field(..., max_length=100)
    model_version: str = Field(..., max_length=20)
    prediction_count: int = Field(0, ge=0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_prediction_time_ms: Optional[float] = None
    cache_hit_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_count: int = Field(0, ge=0)
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    memory_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    disk_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    active_connections: Optional[int] = Field(None, ge=0)

class ModelPerformanceCreate(ModelPerformanceBase):
    pass

class ModelPerformanceResponse(ModelPerformanceBase):
    id: uuid.UUID
    timestamp: datetime

# Feature Importance schemas
class FeatureImportanceBase(BaseSchema):
    feature_name: str = Field(..., max_length=100)
    importance_score: float = Field(..., ge=0.0)
    importance_rank: int = Field(..., ge=1)
    feature_type: FeatureType

class FeatureImportanceCreate(FeatureImportanceBase):
    model_training_id: uuid.UUID

class FeatureImportanceResponse(FeatureImportanceBase):
    id: uuid.UUID
    model_training_id: uuid.UUID
    created_at: datetime

# Audit Log schemas (Simplified - no user authentication)
class AuditLogBase(BaseSchema):
    action: AuditAction
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class AuditLogCreate(AuditLogBase):
    application_id: uuid.UUID

class AuditLogResponse(AuditLogBase):
    id: uuid.UUID
    application_id: uuid.UUID
    timestamp: datetime

# System Config schemas
class SystemConfigBase(BaseSchema):
    config_key: str = Field(..., max_length=100)
    config_value: str
    config_type: ConfigType
    description: Optional[str] = None
    is_active: bool = True

class SystemConfigCreate(SystemConfigBase):
    pass

class SystemConfigUpdate(BaseSchema):
    config_value: Optional[str] = None
    config_type: Optional[ConfigType] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class SystemConfigResponse(SystemConfigBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

# Prediction Cache schemas
class PredictionCacheBase(BaseSchema):
    input_hash: str = Field(..., max_length=64)
    prediction_data: Dict[str, Any]
    expires_at: datetime
    access_count: int = Field(0, ge=0)

class PredictionCacheCreate(PredictionCacheBase):
    pass

class PredictionCacheResponse(PredictionCacheBase):
    id: uuid.UUID
    created_at: datetime
    last_accessed_at: datetime

# User schemas
class UserBase(BaseSchema):
    username: str = Field(..., max_length=50)
    email: str = Field(..., max_length=100)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    role: UserRole = UserRole.user
    is_active: bool = True
    is_verified: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=255)

class UserUpdate(BaseSchema):
    username: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = Field(None, max_length=100)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

class UserResponse(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class UserLogin(BaseSchema):
    username: str
    password: str

class Token(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    user: Optional[UserResponse] = None

# API Response schemas
class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

class ErrorResponse(BaseSchema):
    detail: str
    error_code: Optional[str] = None

class SuccessResponse(BaseSchema):
    message: str
    data: Optional[Any] = None
