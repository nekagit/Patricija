from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import hashlib
import json
from datetime import datetime, timedelta

from database import get_db

from crud import (
    create_prediction, get_prediction, get_predictions_by_application,
    get_predictions_by_model, create_prediction_cache, get_prediction_cache,
    update_cache_access, get_prediction_statistics, cleanup_expired_cache
)
from schemas import (
    PredictionCreate, PredictionResponse, PaginatedResponse
)
from models import Prediction, PredictionCache, CreditApplication

router = APIRouter(prefix="/predictions", tags=["Predictions"])

def hash_input_data(input_data: dict) -> str:
    """Create a hash of input data for caching."""
    data_str = json.dumps(input_data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

@router.post("/", response_model=PredictionResponse)
def create_prediction_endpoint(
    prediction_data: dict,
    db: Session = Depends(get_db)
):
    """Create a new prediction."""
    # Check if we have a cached result
    input_hash = hash_input_data(prediction_data.get("input_features", {}))
    cached_result = get_prediction_cache(db, input_hash)
    
    if cached_result:
        # Update cache access
        update_cache_access(db, cached_result.id)
        
        # Return cached prediction
        cached_prediction_data = cached_result.prediction_data
        return PredictionResponse(
            id=cached_result.id,
            application_id=prediction_data.get("application_id"),
            model_name=cached_prediction_data.get("model_name"),
            model_version=cached_prediction_data.get("model_version"),
            prediction=cached_prediction_data.get("prediction"),
            probability_good=cached_prediction_data.get("probability_good"),
            probability_bad=cached_prediction_data.get("probability_bad"),
            confidence_score=cached_prediction_data.get("confidence_score"),
            risk_category=cached_prediction_data.get("risk_category"),
            input_features=cached_prediction_data.get("input_features"),
            feature_importance=cached_prediction_data.get("feature_importance"),
            shap_values=cached_prediction_data.get("shap_values"),
            lime_explanation=cached_prediction_data.get("lime_explanation"),
            feature_contributions=cached_prediction_data.get("feature_contributions"),
            processing_time_ms=cached_prediction_data.get("processing_time_ms"),
            cache_hit=True,
            input_hash=input_hash,
            prediction_timestamp=cached_result.created_at
        )
    
    # Create new prediction
    prediction_create = PredictionCreate(
        application_id=prediction_data.get("application_id"),
        model_name=prediction_data.get("model_name"),
        model_version=prediction_data.get("model_version"),
        prediction=prediction_data.get("prediction"),
        probability_good=prediction_data.get("probability_good"),
        probability_bad=prediction_data.get("probability_bad"),
        confidence_score=prediction_data.get("confidence_score"),
        risk_category=prediction_data.get("risk_category"),
        input_features=prediction_data.get("input_features"),
        feature_importance=prediction_data.get("feature_importance"),
        shap_values=prediction_data.get("shap_values"),
        lime_explanation=prediction_data.get("lime_explanation"),
        feature_contributions=prediction_data.get("feature_contributions"),
        processing_time_ms=prediction_data.get("processing_time_ms"),
        cache_hit=False,
        input_hash=input_hash
    )
    
    db_prediction = create_prediction(db, prediction_create)
    
    # Cache the result for future use
    cache_data = {
        "model_name": db_prediction.model_name,
        "model_version": db_prediction.model_version,
        "prediction": db_prediction.prediction,
        "probability_good": db_prediction.probability_good,
        "probability_bad": db_prediction.probability_bad,
        "confidence_score": db_prediction.confidence_score,
        "risk_category": db_prediction.risk_category,
        "input_features": db_prediction.input_features,
        "feature_importance": db_prediction.feature_importance,
        "shap_values": db_prediction.shap_values,
        "lime_explanation": db_prediction.lime_explanation,
        "feature_contributions": db_prediction.feature_contributions,
        "processing_time_ms": db_prediction.processing_time_ms
    }
    
    cache_entry = {
        "input_hash": input_hash,
        "prediction_data": cache_data,
        "expires_at": datetime.utcnow() + timedelta(hours=24)
    }
    create_prediction_cache(db, cache_entry)
    
    return db_prediction

@router.get("/", response_model=PaginatedResponse)
def get_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_name: Optional[str] = Query(None),
    application_id: Optional[uuid.UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get predictions with pagination."""
    if application_id:
        predictions = get_predictions_by_application(db, application_id, skip, limit)
    elif model_name:
        predictions = get_predictions_by_model(db, model_name, skip, limit)
    else:
        # Get all predictions
        predictions = db.query(Prediction).offset(skip).limit(limit).all()
    
    total = len(predictions)  # In a real app, you'd get total count separately
    
    return PaginatedResponse(
        items=predictions,
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )

@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction_endpoint(
    prediction_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get a specific prediction."""
    db_prediction = get_prediction(db, prediction_id)
    if not db_prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return db_prediction

@router.get("/application/{application_id}", response_model=List[PredictionResponse])
def get_predictions_for_application(
    application_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get all predictions for a specific application."""
    # Check if application exists
    application = db.query(CreditApplication).filter(
        CreditApplication.id == application_id
    ).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    predictions = get_predictions_by_application(db, application_id)
    return predictions

@router.get("/statistics/summary")
def get_prediction_statistics(
    model_name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get prediction statistics."""
    stats = get_prediction_statistics(db, model_name)
    return stats

@router.delete("/cache/cleanup")
def cleanup_prediction_cache(
    db: Session = Depends(get_db)
):
    """Clean up expired prediction cache entries."""
    expired_count = cleanup_expired_cache(db)
    return {"message": f"Cleaned up {expired_count} expired cache entries"}
