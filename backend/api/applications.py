from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from database import get_db

from crud import (
    create_credit_application, get_credit_application, get_credit_applications,
    update_credit_application, delete_credit_application, create_audit_log,
    get_application_statistics, get_predictions_by_application
)
from schemas import (
    CreditApplicationCreate, CreditApplicationUpdate, CreditApplicationResponse,
    AuditLogCreate, PaginatedResponse
)
from models import CreditApplication

router = APIRouter(prefix="/applications", tags=["Credit Applications"])

@router.post("/", response_model=CreditApplicationResponse)
def create_application(
    application: CreditApplicationCreate,
    db: Session = Depends(get_db)
):
    """Create a new credit application."""
    db_application = create_credit_application(db, application)
    
    # Create audit log
    audit_log = AuditLogCreate(
        application_id=db_application.id,
        action="created",
        new_values=application.dict()
    )
    create_audit_log(db, audit_log)
    
    return db_application

@router.get("/", response_model=PaginatedResponse)
def get_applications(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get credit applications with pagination."""
    applications = get_credit_applications(db, status=status, skip=skip, limit=limit)
    total = len(applications)  # In a real app, you'd get total count separately
    
    return PaginatedResponse(
        items=applications,
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )

@router.get("/{application_id}", response_model=CreditApplicationResponse)
def get_application(
    application_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get a specific credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    return db_application

@router.put("/{application_id}", response_model=CreditApplicationResponse)
def update_application(
    application_id: uuid.UUID,
    application_update: CreditApplicationUpdate,
    db: Session = Depends(get_db)
):
    """Update a credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Store old values for audit log
    old_values = {
        "status": db_application.status,
        "notes": db_application.notes
    }
    
    updated_application = update_credit_application(db, application_id, application_update)
    
    # Create audit log
    audit_log = AuditLogCreate(
        application_id=application_id,
        action="updated",
        old_values=old_values,
        new_values=application_update.dict(exclude_unset=True)
    )
    create_audit_log(db, audit_log)
    
    return updated_application

@router.delete("/{application_id}")
def delete_application(
    application_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete a credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    success = delete_credit_application(db, application_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete application")
    
    return {"message": "Application deleted successfully"}

@router.post("/{application_id}/review")
def review_application(
    application_id: uuid.UUID,
    review_data: dict,
    db: Session = Depends(get_db)
):
    """Review a credit application."""
    db_application = get_credit_application(db, application_id)
    if not db_application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Update application with review
    update_data = CreditApplicationUpdate(
        status=review_data.get("status"),
        notes=review_data.get("notes")
    )
    
    updated_application = update_credit_application(db, application_id, update_data)
    
    # Create audit log
    audit_log = AuditLogCreate(
        application_id=application_id,
        action="reviewed",
        new_values=review_data
    )
    create_audit_log(db, audit_log)
    
    return {"message": "Application reviewed successfully", "application": updated_application}

@router.get("/statistics/summary")
def get_application_statistics(
    db: Session = Depends(get_db)
):
    """Get application statistics."""
    stats = get_application_statistics(db)
    return stats

@router.get("/history")
def get_application_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get credit application history with predictions."""
    try:
        # Get applications with their predictions
        applications = get_credit_applications(db, skip=skip, limit=limit)
        
        history_data = []
        for app in applications:
            # Get predictions for this application
            predictions = get_predictions_by_application(db, app.id)
            
            app_data = {
                "id": str(app.id),
                "application_number": app.application_number,
                "status": app.status,
                "created_at": app.created_at.isoformat() if app.created_at else None,
                "submitted_at": app.submitted_at.isoformat() if app.submitted_at else None,
                "application_data": {
                    "person_age": app.person_age,
                    "person_income": app.person_income,
                    "person_home_ownership": app.person_home_ownership,
                    "person_emp_length": app.person_emp_length,
                    "cb_person_cred_hist_length": app.cb_person_cred_hist_length,
                    "cb_person_default_on_file": app.cb_person_default_on_file,
                    "loan_intent": app.loan_intent,
                    "loan_grade": app.loan_grade,
                    "loan_amnt": app.loan_amnt,
                    "loan_int_rate": app.loan_int_rate,
                    "loan_percent_income": app.loan_percent_income
                },
                "predictions": []
            }
            
            for pred in predictions:
                pred_data = {
                    "id": str(pred.id),
                    "model_name": pred.model_name,
                    "model_version": pred.model_version,
                    "prediction": pred.prediction,
                    "probability_good": pred.probability_good,
                    "probability_bad": pred.probability_bad,
                    "confidence_score": pred.confidence_score,
                    "risk_category": pred.risk_category,
                    "prediction_timestamp": pred.prediction_timestamp.isoformat() if pred.prediction_timestamp else None,
                    "processing_time_ms": pred.processing_time_ms,
                    "feature_importance": pred.feature_importance,
                    "shap_values": pred.shap_values
                }
                app_data["predictions"].append(pred_data)
            
            history_data.append(app_data)
        
        # If no real data, return demo data
        if not history_data:
            # Call the demo history endpoint
            demo_response = get_demo_history()
            return demo_response
        
        total = len(history_data)
        
        return {
            "items": history_data,
            "total": total,
            "page": skip // limit + 1,
            "size": limit,
            "pages": (total + limit - 1) // limit
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get application history: {str(e)}")

@router.get("/demo-data")
def get_demo_data():
    """Get demo data for when backend is not connected or has no real data."""
    try:
        # Try to load demo data from CSV file
        demo_data_path = Path(__file__).parent.parent.parent / "frontend" / "data" / "credit_risk_dataset.csv"
        if not demo_data_path.exists():
            demo_data_path = Path(__file__).parent.parent.parent / "frontend" / "data" / "sample_credit_data.csv"
        
        if demo_data_path.exists():
            df = pd.read_csv(demo_data_path)
            
            # Convert DataFrame to list of dictionaries
            demo_data = []
            for _, row in df.head(100).iterrows():  # Limit to 100 records for demo
                # Convert row to application format
                application_data = {
                    "id": str(uuid.uuid4()),
                    "application_number": f"DEMO-{len(demo_data)+1:04d}",
                    "status": "approved" if np.random.random() > 0.3 else "rejected",
                    "person_age": int(row.get('person_age', np.random.randint(25, 65))),
                    "person_income": float(row.get('person_income', np.random.randint(30000, 120000))),
                    "person_home_ownership": row.get('person_home_ownership', np.random.choice(['RENT', 'OWN', 'MORTGAGE'])),
                    "person_emp_length": float(row.get('person_emp_length', np.random.uniform(1, 20))),
                    "cb_person_cred_hist_length": int(row.get('cb_person_cred_hist_length', np.random.randint(1, 15))),
                    "cb_person_default_on_file": row.get('cb_person_default_on_file', np.random.choice(['Y', 'N'])),
                    "loan_intent": row.get('loan_intent', np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])),
                    "loan_grade": row.get('loan_grade', np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'])),
                    "loan_amnt": float(row.get('loan_amnt', np.random.randint(5000, 50000))),
                    "loan_int_rate": float(row.get('loan_int_rate', np.random.uniform(5, 25))),
                    "loan_percent_income": float(row.get('loan_percent_income', np.random.uniform(0.1, 0.8))),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "submitted_at": datetime.now().isoformat()
                }
                demo_data.append(application_data)
            
            return {
                "items": demo_data,
                "total": len(demo_data),
                "page": 1,
                "size": len(demo_data),
                "pages": 1,
                "is_demo_data": True
            }
        else:
            # Generate synthetic demo data if no CSV file exists
            demo_data = []
            for i in range(50):
                application_data = {
                    "id": str(uuid.uuid4()),
                    "application_number": f"DEMO-{i+1:04d}",
                    "status": "approved" if np.random.random() > 0.3 else "rejected",
                    "person_age": np.random.randint(25, 65),
                    "person_income": np.random.randint(30000, 120000),
                    "person_home_ownership": np.random.choice(['RENT', 'OWN', 'MORTGAGE']),
                    "person_emp_length": np.random.uniform(1, 20),
                    "cb_person_cred_hist_length": np.random.randint(1, 15),
                    "cb_person_default_on_file": np.random.choice(['Y', 'N']),
                    "loan_intent": np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']),
                    "loan_grade": np.random.choice(['A', 'B', 'C', 'D', 'E', 'F']),
                    "loan_amnt": np.random.randint(5000, 50000),
                    "loan_int_rate": np.random.uniform(5, 25),
                    "loan_percent_income": np.random.uniform(0.1, 0.8),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "submitted_at": datetime.now().isoformat()
                }
                demo_data.append(application_data)
            
            return {
                "items": demo_data,
                "total": len(demo_data),
                "page": 1,
                "size": len(demo_data),
                "pages": 1,
                "is_demo_data": True
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate demo data: {str(e)}")

@router.get("/demo-history")
def get_demo_history():
    """Get demo history data with predictions for when backend has no real data."""
    try:
        demo_history = []
        
        # Generate demo applications with predictions
        for i in range(20):
            # Generate application data
            person_age = np.random.randint(25, 65)
            person_income = np.random.randint(30000, 120000)
            person_home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'])
            person_emp_length = np.random.uniform(1, 20)
            cb_person_cred_hist_length = np.random.randint(1, 15)
            cb_person_default_on_file = np.random.choice(['Y', 'N'])
            loan_intent = np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
            loan_grade = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'])
            loan_amnt = np.random.randint(5000, 50000)
            loan_int_rate = np.random.uniform(5, 25)
            loan_percent_income = np.random.uniform(0.1, 0.8)
            
            # Determine prediction based on some logic
            is_good_credit = (
                person_income > 60000 and
                person_emp_length > 3 and
                cb_person_default_on_file == 'N' and
                loan_percent_income < 0.5
            )
            
            prediction = "Good" if is_good_credit else "Bad"
            probability_good = 0.85 if is_good_credit else 0.25
            probability_bad = 1 - probability_good
            
            # Create demo application
            app_data = {
                "id": str(uuid.uuid4()),
                "application_number": f"DEMO-HIST-{i+1:04d}",
                "status": "approved" if is_good_credit else "rejected",
                "created_at": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                "submitted_at": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                "application_data": {
                    "person_age": person_age,
                    "person_income": person_income,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "cb_person_cred_hist_length": cb_person_cred_hist_length,
                    "cb_person_default_on_file": cb_person_default_on_file,
                    "loan_intent": loan_intent,
                    "loan_grade": loan_grade,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": loan_int_rate,
                    "loan_percent_income": loan_percent_income
                },
                "predictions": [
                    {
                        "id": str(uuid.uuid4()),
                        "model_name": np.random.choice(['Random Forest', 'Gradient Boosting', 'Neural Network', 'Support Vector Machine']),
                        "model_version": "1.0.0",
                        "prediction": prediction,
                        "probability_good": probability_good,
                        "probability_bad": probability_bad,
                        "confidence_score": probability_good if prediction == "Good" else probability_bad,
                        "risk_category": "low" if is_good_credit else "high",
                        "prediction_timestamp": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                        "processing_time_ms": np.random.randint(50, 500),
                        "feature_importance": {
                            "person_income": np.random.uniform(0.1, 0.3),
                            "loan_percent_income": np.random.uniform(0.1, 0.25),
                            "person_emp_length": np.random.uniform(0.05, 0.15),
                            "cb_person_default_on_file": np.random.uniform(0.1, 0.2),
                            "loan_grade": np.random.uniform(0.05, 0.15),
                            "person_age": np.random.uniform(0.05, 0.1)
                        },
                        "shap_values": None
                    }
                ]
            }
            
            demo_history.append(app_data)
        
        return {
            "items": demo_history,
            "total": len(demo_history),
            "page": 1,
            "size": len(demo_history),
            "pages": 1,
            "is_demo_data": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate demo history: {str(e)}")
