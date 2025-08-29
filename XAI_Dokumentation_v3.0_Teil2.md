# XAI - Prozess-basierte Code-Analyse v3.0
## Umfassende Dokumentation (35 Seiten) - Teil 2

**Fortsetzung von Teil 1**

---

## 3. Datenaufbereitung und Feature Engineering

### 3.1 Datenquellen und Datensätze

**Kaggle Credit Risk Dataset:**

Das verwendete Dataset stammt aus dem Kaggle Credit Risk Dataset und enthält echte Kreditanträge mit anonymisierten persönlichen Daten. Die Datenqualität und -struktur sind entscheidend für die Modell-Performance.

**Dataset-Statistiken:**
```python
# Dataset-Übersicht
dataset_stats = {
    "total_records": 32581,
    "features": 12,
    "target_variable": "loan_status",
    "missing_values": 0.15,  # 15% fehlende Werte
    "class_imbalance": 4.2,  # 4.2:1 Verhältnis
    "data_size_mb": 2.8,
    "last_updated": "2023-12-01"
}

# Feature-Übersicht
features_overview = {
    "numerical_features": [
        "person_age",           # 18-100 Jahre
        "person_income",        # 4K-600K USD
        "person_emp_length",    # 0-43 Jahre
        "loan_amnt",           # 500-35000 USD
        "loan_int_rate",       # 5.42-23.22%
        "loan_percent_income", # 0.0-0.9
        "cb_person_cred_hist_length"  # 0-30 Jahre
    ],
    "categorical_features": [
        "person_home_ownership",  # RENT, MORTGAGE, OWN, OTHER
        "loan_intent",           # PERSONAL, EDUCATION, MEDICAL, etc.
        "loan_grade",            # A, B, C, D, E, F, G
        "cb_person_default_on_file",  # Y, N
        "cb_person_cred_hist_length"  # 0-30
    ],
    "target_feature": "loan_status"  # 0=Default, 1=Good
}
```

**Datenqualitätsanalyse:**

```python
def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Umfassende Datenqualitätsanalyse"""
    
    quality_report = {
        "missing_values": {},
        "data_types": {},
        "value_ranges": {},
        "outliers": {},
        "duplicates": 0,
        "data_consistency": {}
    }
    
    # Missing Values Analysis
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        quality_report["missing_values"][column] = {
            "count": missing_count,
            "percentage": missing_percentage,
            "severity": "high" if missing_percentage > 20 else "medium" if missing_percentage > 5 else "low"
        }
    
    # Data Type Analysis
    for column in df.columns:
        quality_report["data_types"][column] = {
            "current_type": str(df[column].dtype),
            "expected_type": get_expected_type(column),
            "conversion_needed": df[column].dtype != get_expected_type(column)
        }
    
    # Value Range Analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        quality_report["value_ranges"][column] = {
            "min": df[column].min(),
            "max": df[column].max(),
            "mean": df[column].mean(),
            "median": df[column].median(),
            "std": df[column].std()
        }
    
    # Outlier Detection
    for column in numeric_columns:
        outliers = detect_outliers_iqr(df[column])
        quality_report["outliers"][column] = {
            "count": outliers.sum(),
            "percentage": (outliers.sum() / len(df)) * 100
        }
    
    # Duplicate Analysis
    quality_report["duplicates"] = df.duplicated().sum()
    
    return quality_report
```

### 3.2 Datenvalidierung und Qualitätssicherung

**Umfassende Validierungsstrategie:**

```python
class DataValidator:
    """Umfassende Datenvalidierung für Kreditrisiko-Daten"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.error_log = []
        self.warning_log = []
    
    def _load_validation_rules(self) -> dict:
        """Lädt Validierungsregeln für alle Features"""
        return {
            "person_age": {
                "type": "integer",
                "range": (18, 100),
                "required": True,
                "business_logic": "Alter muss realistisch für Kreditantrag sein"
            },
            "person_income": {
                "type": "float",
                "range": (0, 1000000),
                "required": True,
                "business_logic": "Einkommen muss positiv und realistisch sein"
            },
            "loan_amnt": {
                "type": "float",
                "range": (500, 100000),
                "required": True,
                "business_logic": "Kreditsumme muss im üblichen Bereich liegen"
            },
            "person_home_ownership": {
                "type": "categorical",
                "allowed_values": ["RENT", "MORTGAGE", "OWN", "OTHER"],
                "required": True,
                "business_logic": "Hausbesitz-Status muss gültig sein"
            },
            "loan_intent": {
                "type": "categorical",
                "allowed_values": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
                                 "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                "required": True,
                "business_logic": "Kreditzweck muss gültig sein"
            }
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> dict:
        """Führt umfassende Datenvalidierung durch"""
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
            "recommendations": []
        }
        
        # 1. Schema Validation
        schema_validation = self._validate_schema(df)
        validation_results["errors"].extend(schema_validation["errors"])
        validation_results["warnings"].extend(schema_validation["warnings"])
        
        # 2. Data Type Validation
        type_validation = self._validate_data_types(df)
        validation_results["errors"].extend(type_validation["errors"])
        
        # 3. Range Validation
        range_validation = self._validate_ranges(df)
        validation_results["errors"].extend(range_validation["errors"])
        validation_results["warnings"].extend(range_validation["warnings"])
        
        # 4. Business Logic Validation
        business_validation = self._validate_business_logic(df)
        validation_results["errors"].extend(business_validation["errors"])
        validation_results["warnings"].extend(business_validation["warnings"])
        
        # 5. Consistency Validation
        consistency_validation = self._validate_consistency(df)
        validation_results["errors"].extend(consistency_validation["errors"])
        
        # Update validation status
        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(validation_results)
        
        return validation_results
    
    def _validate_schema(self, df: pd.DataFrame) -> dict:
        """Validiert das Datenbankschema"""
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = [col for col, rules in self.validation_rules.items() 
                          if rules.get("required", False)]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Fehlende erforderliche Spalten: {missing_columns}")
        
        # Check for unexpected columns
        expected_columns = list(self.validation_rules.keys())
        unexpected_columns = [col for col in df.columns if col not in expected_columns]
        if unexpected_columns:
            warnings.append(f"Unerwartete Spalten gefunden: {unexpected_columns}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_business_logic(self, df: pd.DataFrame) -> dict:
        """Validiert Business Logic Regeln"""
        errors = []
        warnings = []
        
        # Rule 1: Kreditsumme sollte nicht mehr als 5x Jahreseinkommen sein
        if 'loan_amnt' in df.columns and 'person_income' in df.columns:
            high_ratio_mask = df['loan_amnt'] > df['person_income'] * 5
            high_ratio_count = high_ratio_mask.sum()
            if high_ratio_count > 0:
                warnings.append(f"{high_ratio_count} Anträge haben Kreditsumme > 5x Einkommen")
        
        # Rule 2: Alter sollte konsistent mit Beschäftigungsdauer sein
        if 'person_age' in df.columns and 'person_emp_length' in df.columns:
            impossible_emp_mask = df['person_emp_length'] > (df['person_age'] - 18)
            impossible_count = impossible_emp_mask.sum()
            if impossible_count > 0:
                errors.append(f"{impossible_count} Anträge haben unmögliche Beschäftigungsdauer")
        
        # Rule 3: Kreditzinsen sollten im realistischen Bereich liegen
        if 'loan_int_rate' in df.columns:
            high_rate_mask = df['loan_int_rate'] > 25
            low_rate_mask = df['loan_int_rate'] < 3
            if high_rate_mask.sum() > 0:
                warnings.append(f"{high_rate_mask.sum()} Anträge haben sehr hohe Zinsen (>25%)")
            if low_rate_mask.sum() > 0:
                warnings.append(f"{low_rate_mask.sum()} Anträge haben sehr niedrige Zinsen (<3%)")
        
        return {"errors": errors, "warnings": warnings}
```

### 3.3 Feature Engineering Pipeline

**Erweiterte Feature Engineering Strategien:**

```python
class AdvancedFeatureEngineer:
    """Erweiterte Feature Engineering Pipeline für Kreditrisiko-Analysen"""
    
    def __init__(self):
        self.feature_config = self._load_feature_config()
        self.engineered_features = []
    
    def _load_feature_config(self) -> dict:
        """Lädt Feature Engineering Konfiguration"""
        return {
            "numerical_features": {
                "loan_percent_income": {
                    "description": "Verhältnis Kreditsumme zu Jahreseinkommen",
                    "formula": "loan_amnt / person_income",
                    "clipping": (0, 1),
                    "importance": "high"
                },
                "income_per_age": {
                    "description": "Einkommen pro Lebensjahr",
                    "formula": "person_income / person_age",
                    "clipping": None,
                    "importance": "medium"
                },
                "emp_per_age": {
                    "description": "Beschäftigungsdauer pro Lebensjahr",
                    "formula": "person_emp_length / person_age",
                    "clipping": (0, 1),
                    "importance": "medium"
                },
                "debt_to_income_ratio": {
                    "description": "Schulden-zu-Einkommen Verhältnis",
                    "formula": "loan_amnt / (person_income * 12)",
                    "clipping": (0, 0.5),
                    "importance": "high"
                }
            },
            "categorical_features": {
                "age_group": {
                    "bins": [0, 25, 35, 45, 55, 100],
                    "labels": ["18-25", "26-35", "36-45", "46-55", "55+"],
                    "importance": "medium"
                },
                "income_group": {
                    "bins": [0, 30000, 50000, 75000, 100000, float('inf')],
                    "labels": ["Niedrig", "Mittel-Niedrig", "Mittel", "Mittel-Hoch", "Hoch"],
                    "importance": "high"
                },
                "loan_size_category": {
                    "bins": [0, 5000, 10000, 20000, float('inf')],
                    "labels": ["Klein", "Mittel", "Groß", "Sehr Groß"],
                    "importance": "medium"
                }
            },
            "interaction_features": {
                "age_income_interaction": {
                    "description": "Interaktion zwischen Alter und Einkommen",
                    "formula": "person_age * person_income / 1000000",
                    "importance": "medium"
                },
                "emp_income_interaction": {
                    "description": "Interaktion zwischen Beschäftigungsdauer und Einkommen",
                    "formula": "person_emp_length * person_income / 1000000",
                    "importance": "low"
                }
            }
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Führt erweiterte Feature Engineering durch"""
        
        df_engineered = df.copy()
        
        # 1. Numerische Features
        df_engineered = self._create_numerical_features(df_engineered)
        
        # 2. Kategorische Features
        df_engineered = self._create_categorical_features(df_engineered)
        
        # 3. Interaktions-Features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # 4. Polynomische Features
        df_engineered = self._create_polynomial_features(df_engineered)
        
        # 5. Statistische Features
        df_engineered = self._create_statistical_features(df_engineered)
        
        # 6. Domain-spezifische Features
        df_engineered = self._create_domain_features(df_engineered)
        
        return df_engineered
    
    def _create_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt numerische abgeleitete Features"""
        
        for feature_name, config in self.feature_config["numerical_features"].items():
            try:
                # Berechne Feature basierend auf Formel
                if config["formula"] == "loan_amnt / person_income":
                    df[feature_name] = df['loan_amnt'] / df['person_income']
                elif config["formula"] == "person_income / person_age":
                    df[feature_name] = df['person_income'] / df['person_age']
                elif config["formula"] == "person_emp_length / person_age":
                    df[feature_name] = df['person_emp_length'] / df['person_age']
                elif config["formula"] == "loan_amnt / (person_income * 12)":
                    df[feature_name] = df['loan_amnt'] / (df['person_income'] * 12)
                
                # Clipping anwenden
                if config["clipping"]:
                    min_val, max_val = config["clipping"]
                    df[feature_name] = df[feature_name].clip(min_val, max_val)
                
                # NaN-Werte behandeln
                df[feature_name] = df[feature_name].fillna(df[feature_name].median())
                
                self.engineered_features.append(feature_name)
                
            except Exception as e:
                logger.warning(f"Fehler beim Erstellen von {feature_name}: {e}")
        
        return df
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt domänenspezifische Features für Kreditrisiko"""
        
        # 1. Kreditrisiko-Score (vereinfacht)
        if all(col in df.columns for col in ['person_age', 'person_income', 'loan_amnt']):
            df['credit_risk_score'] = (
                (df['person_age'] / 100) * 0.3 +
                (df['person_income'] / 100000) * 0.4 +
                (1 - df['loan_amnt'] / df['person_income']) * 0.3
            )
            self.engineered_features.append('credit_risk_score')
        
        # 2. Finanzielle Stabilität
        if all(col in df.columns for col in ['person_emp_length', 'person_income']):
            df['financial_stability'] = (
                (df['person_emp_length'] / 20) * 0.6 +
                (df['person_income'] / 100000) * 0.4
            )
            self.engineered_features.append('financial_stability')
        
        # 3. Kreditwürdigkeit basierend auf Hausbesitz
        if 'person_home_ownership' in df.columns:
            ownership_scores = {
                'OWN': 1.0,
                'MORTGAGE': 0.8,
                'RENT': 0.5,
                'OTHER': 0.3
            }
            df['ownership_score'] = df['person_home_ownership'].map(ownership_scores)
            self.engineered_features.append('ownership_score')
        
        return df
    
    def get_feature_importance_report(self) -> dict:
        """Erstellt Bericht über erstellte Features"""
        
        report = {
            "total_features_created": len(self.engineered_features),
            "feature_categories": {
                "numerical": [],
                "categorical": [],
                "interaction": [],
                "domain_specific": []
            },
            "importance_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        for feature in self.engineered_features:
            # Kategorisiere Features
            if feature in self.feature_config["numerical_features"]:
                report["feature_categories"]["numerical"].append(feature)
                importance = self.feature_config["numerical_features"][feature]["importance"]
                report["importance_distribution"][importance] += 1
        
        return report
```

### 3.4 Outlier Detection und Behandlung

**Erweiterte Outlier Detection Strategien:**

```python
class AdvancedOutlierDetector:
    """Erweiterte Outlier Detection mit multiplen Methoden"""
    
    def __init__(self):
        self.detection_methods = {
            "iqr": self._detect_outliers_iqr,
            "zscore": self._detect_outliers_zscore,
            "isolation_forest": self._detect_outliers_isolation_forest,
            "local_outlier_factor": self._detect_outliers_lof,
            "elliptic_envelope": self._detect_outliers_elliptic_envelope
        }
        self.treatment_methods = {
            "median_imputation": self._treat_outliers_median,
            "mean_imputation": self._treat_outliers_mean,
            "winsorization": self._treat_outliers_winsorization,
            "removal": self._treat_outliers_removal,
            "capping": self._treat_outliers_capping
        }
    
    def detect_outliers_comprehensive(self, df: pd.DataFrame, 
                                    methods: List[str] = None) -> dict:
        """Führt umfassende Outlier Detection durch"""
        
        if methods is None:
            methods = ["iqr", "zscore", "isolation_forest"]
        
        results = {
            "outlier_summary": {},
            "method_comparison": {},
            "recommendations": [],
            "treated_data": df.copy()
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            column_results = {}
            
            for method in methods:
                if method in self.detection_methods:
                    outliers = self.detection_methods[method](df[column])
                    outlier_count = outliers.sum()
                    outlier_percentage = (outlier_count / len(df)) * 100
                    
                    column_results[method] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_percentage,
                        "outlier_indices": outliers[outliers].index.tolist()
                    }
            
            results["outlier_summary"][column] = column_results
            
            # Empfehlung basierend auf Konsistenz zwischen Methoden
            recommendations = self._generate_outlier_recommendations(column_results)
            results["recommendations"].extend(recommendations)
        
        return results
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> pd.Series:
        """Isolation Forest für Outlier Detection"""
        from sklearn.ensemble import IsolationForest
        
        # Reshape für sklearn
        X = series.values.reshape(-1, 1)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Erwarteter Anteil an Outliers
            random_state=42,
            n_estimators=100
        )
        
        # Predict (-1 für Outliers, 1 für normale Werte)
        predictions = iso_forest.fit_predict(X)
        
        # Konvertiere zu boolean (True für Outliers)
        return pd.Series(predictions == -1, index=series.index)
    
    def _detect_outliers_lof(self, series: pd.Series) -> pd.Series:
        """Local Outlier Factor für Outlier Detection"""
        from sklearn.neighbors import LocalOutlierFactor
        
        # Reshape für sklearn
        X = series.values.reshape(-1, 1)
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(
            contamination=0.1,
            n_neighbors=20,
            metric='euclidean'
        )
        
        # Predict (-1 für Outliers, 1 für normale Werte)
        predictions = lof.fit_predict(X)
        
        # Konvertiere zu boolean (True für Outliers)
        return pd.Series(predictions == -1, index=series.index)
    
    def _treat_outliers_adaptive(self, df: pd.DataFrame, 
                               outlier_results: dict) -> pd.DataFrame:
        """Adaptive Outlier-Behandlung basierend auf Datenanalyse"""
        
        df_treated = df.copy()
        
        for column, results in outlier_results["outlier_summary"].items():
            
            # Bestimme beste Behandlungsmethode
            treatment_method = self._select_treatment_method(column, results)
            
            # Wende Behandlung an
            if treatment_method in self.treatment_methods:
                df_treated = self.treatment_methods[treatment_method](df_treated, column, results)
            
            logger.info(f"Outlier-Behandlung für {column}: {treatment_method}")
        
        return df_treated
    
    def _select_treatment_method(self, column: str, results: dict) -> str:
        """Wählt beste Behandlungsmethode basierend auf Datenanalyse"""
        
        # Analysiere Outlier-Verteilung
        outlier_percentages = [results[method]["outlier_percentage"] 
                             for method in results.keys()]
        
        avg_outlier_percentage = np.mean(outlier_percentages)
        outlier_consistency = np.std(outlier_percentages)
        
        # Entscheidungslogik
        if avg_outlier_percentage < 5:
            return "capping"  # Wenige Outliers -> Capping
        elif avg_outlier_percentage < 15:
            return "winsorization"  # Moderate Outliers -> Winsorization
        elif outlier_consistency < 5:
            return "median_imputation"  # Konsistente Outliers -> Median
        else:
            return "removal"  # Inkonsistente Outliers -> Entfernung
    
    def _treat_outliers_winsorization(self, df: pd.DataFrame, 
                                    column: str, results: dict) -> pd.DataFrame:
        """Winsorization für Outlier-Behandlung"""
        
        # Berechne 1. und 99. Perzentil
        q1 = df[column].quantile(0.01)
        q99 = df[column].quantile(0.99)
        
        # Winsorize
        df[column] = df[column].clip(lower=q1, upper=q99)
        
        return df
    
    def _treat_outliers_capping(self, df: pd.DataFrame, 
                              column: str, results: dict) -> pd.DataFrame:
        """Capping für Outlier-Behandlung"""
        
        # Verwende IQR-Methode für Grenzen
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap Outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
```

---

**Fortsetzung folgt in Teil 3...**
