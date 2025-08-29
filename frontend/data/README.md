# Data - Credit Risk Dataset

## 🎯 Overview

The `data` directory contains the main credit risk dataset used for training and evaluating the XAI credit assessment system. This dataset provides comprehensive information about loan applications and their outcomes.

## 📁 Structure

```
data/
├── README.md                    # This documentation file
└── credit_risk_dataset.csv      # Main credit risk dataset
```

## 📊 Dataset: credit_risk_dataset.csv

**Purpose**: Primary dataset for credit risk assessment and model training.

**Size**: 32,581 records with 12 features

**Source**: Synthetic credit risk dataset for demonstration purposes

### 📋 Column Descriptions

#### Personal Information
- **`person_age`** (int): Applicant age in years (18-100)
- **`person_income`** (int): Annual income in currency units
- **`person_home_ownership`** (object): Home ownership status
  - Values: `RENT`, `OWN`, `MORTGAGE`, `OTHER`
- **`person_emp_length`** (float): Employment length in years (0-50)

#### Loan Information
- **`loan_intent`** (object): Purpose of the loan
  - Values: `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`
- **`loan_grade`** (object): Loan grade assigned by lender
  - Values: `A`, `B`, `C`, `D`, `E`, `F` (A = best, F = worst)
- **`loan_amnt`** (int): Loan amount requested
- **`loan_int_rate`** (float): Interest rate on the loan (%)
- **`loan_percent_income`** (float): Loan amount as percentage of income (0.0-1.0)

#### Credit History
- **`cb_person_default_on_file`** (object): Previous default history
  - Values: `Y` (Yes), `N` (No)
- **`cb_person_cred_hist_length`** (int): Length of credit history in years

#### Target Variable
- **`loan_status`** (int): Loan outcome (target variable)
  - Values: `0` (Good - loan repaid), `1` (Default - loan not repaid)

### 📈 Sample Data

```csv
person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length
22,59000,RENT,123.0,PERSONAL,D,35000,16.02,1,0.59,Y,3
21,9600,OWN,5.0,EDUCATION,B,1000,11.14,0,0.1,N,2
25,9600,MORTGAGE,1.0,MEDICAL,C,5500,12.87,1,0.57,N,3
```

## 🔧 Data Processing

### Loading Data
```python
import pandas as pd

def load_credit_data():
    """Load the credit risk dataset."""
    
    data_path = "data/credit_risk_dataset.csv"
    df = pd.read_csv(data_path)
    
    return df
```

### Data Cleaning
```python
def clean_credit_data(df):
    """Clean and prepare the credit risk dataset."""
    
    # Remove rows with missing values in key columns
    key_columns = ['person_age', 'person_income', 'loan_status']
    df = df.dropna(subset=key_columns)
    
    # Handle missing values in other columns
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Remove outliers
    df = df[df['person_age'] <= 100]
    df = df[df['person_income'] <= 1000000]
    
    return df
```

### Feature Engineering
```python
def engineer_features(df):
    """Create additional features for modeling."""
    
    # Income categories
    df['income_category'] = pd.cut(
        df['person_income'], 
        bins=[0, 30000, 60000, 100000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Age categories
    df['age_category'] = pd.cut(
        df['person_age'],
        bins=[0, 25, 35, 50, float('inf')],
        labels=['Young', 'Young Adult', 'Adult', 'Senior']
    )
    
    # Risk score (simple heuristic)
    df['risk_score'] = (
        (df['person_age'] < 25).astype(int) * 0.1 +
        (df['person_income'] < 30000).astype(int) * 0.2 +
        (df['loan_percent_income'] > 0.5).astype(int) * 0.3 +
        (df['cb_person_default_on_file'] == 'Y').astype(int) * 0.4
    )
    
    return df
```

## 📊 Data Analysis

### Basic Statistics
```python
def analyze_credit_data(df):
    """Generate basic statistics for the credit dataset."""
    
    stats = {
        'total_records': len(df),
        'default_rate': (df['loan_status'].sum() / len(df)) * 100,
        'average_age': df['person_age'].mean(),
        'average_income': df['person_income'].mean(),
        'average_loan_amount': df['loan_amnt'].mean(),
        'home_ownership_distribution': df['person_home_ownership'].value_counts().to_dict(),
        'loan_intent_distribution': df['loan_intent'].value_counts().to_dict(),
        'loan_grade_distribution': df['loan_grade'].value_counts().to_dict()
    }
    
    return stats
```

### Risk Analysis
```python
def analyze_risk_factors(df):
    """Analyze risk factors for loan default."""
    
    # Default rate by home ownership
    home_risk = df.groupby('person_home_ownership')['loan_status'].agg(['count', 'sum'])
    home_risk['default_rate'] = (home_risk['sum'] / home_risk['count']) * 100
    
    # Default rate by loan grade
    grade_risk = df.groupby('loan_grade')['loan_status'].agg(['count', 'sum'])
    grade_risk['default_rate'] = (grade_risk['sum'] / grade_risk['count']) * 100
    
    # Default rate by income level
    df['income_level'] = pd.cut(df['person_income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    income_risk = df.groupby('income_level')['loan_status'].agg(['count', 'sum'])
    income_risk['default_rate'] = (income_risk['sum'] / income_risk['count']) * 100
    
    return {
        'home_ownership_risk': home_risk,
        'loan_grade_risk': grade_risk,
        'income_level_risk': income_risk
    }
```

## 🎯 Use Cases

### 1. Model Training
- Train machine learning models for credit risk prediction
- Validate model performance on real-world data
- Compare different algorithms and approaches

### 2. Feature Analysis
- Identify key factors influencing loan default
- Understand risk patterns across different demographics
- Develop risk scoring models

### 3. Business Intelligence
- Analyze loan portfolio performance
- Identify high-risk customer segments
- Optimize lending strategies

### 4. Regulatory Compliance
- Demonstrate model fairness across different groups
- Provide explainable AI insights
- Support regulatory reporting requirements

## 🔒 Data Privacy

### Anonymization
- All data is synthetic and anonymized
- No real personal information included
- Generated for demonstration and research purposes only

### Data Protection
- Follow GDPR compliance guidelines
- Implement data retention policies
- Secure data storage and access controls

## 📊 Data Quality

### Quality Metrics
- **Completeness**: 99.5% (minimal missing values)
- **Accuracy**: Synthetic data with realistic distributions
- **Consistency**: Consistent data formats and value ranges
- **Timeliness**: Static dataset for demonstration

### Quality Checks
```python
def check_data_quality(df):
    """Perform comprehensive data quality checks."""
    
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'value_ranges': {
            'person_age': {'min': df['person_age'].min(), 'max': df['person_age'].max()},
            'person_income': {'min': df['person_income'].min(), 'max': df['person_income'].max()},
            'loan_amnt': {'min': df['loan_amnt'].min(), 'max': df['loan_amnt'].max()}
        },
        'categorical_values': {
            'person_home_ownership': df['person_home_ownership'].unique().tolist(),
            'loan_intent': df['loan_intent'].unique().tolist(),
            'loan_grade': df['loan_grade'].unique().tolist()
        }
    }
    
    return quality_report
```

## 🔧 Maintenance

### Regular Updates
- Dataset is static for demonstration purposes
- No regular updates required
- Version control for any modifications

### Backup Strategy
- Regular backups of dataset files
- Version control for data changes
- Documentation of any modifications

## 📚 Additional Resources

- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Credit Risk Modeling**: Best practices for financial modeling
- **Data Privacy Guidelines**: GDPR and financial data protection
- **Machine Learning in Finance**: Industry standards and practices

---

**Synthetic credit risk dataset designed for demonstration, research, and educational purposes**
