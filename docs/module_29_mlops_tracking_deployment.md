# Module 29: MLOps: Experiment Tracking & Model Deployment

## 1. The Physical Intuition: From Jupyter Notebooks to Production REST APIs

Imagine building a custom sports car engine in your personal home garage. You tinker with carburetors, try different oil blends, and get the engine to run once on a Saturday afternoon.

Now, someone asks you: *"Can you manufacture 100,000 identical units of this engine per month, guarantee 99.99% uptime, track the exact origin of every bolt, and instantly push software over-the-air updates without stopping the assembly line?"*

That is the difference between a **Jupyter Notebook** and **Production MLOps**.

```
Jupyter Notebook:    Experimental scratchpad (untracked parameters, manual runs, zero reproducibility).
MLOps Pipeline:      Automated software factory (MLflow experiment tracking + FastAPI microservice containerization).
```

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. MLflow Experiment Tracking
- Tracks parameters, metrics, code versions, and trained model artifacts into an audit-logged sqlite database.

---

### 2. FastAPI Model Serving Microservice
- Wraps trained models in asynchronous REST API endpoints with **Pydantic** input schema validation.

---

## 3. Practical Implementation: Production MLOps Stack

Let's write a complete Python script demonstrating MLflow experiment tracking followed by a production FastAPI microservice definition:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# PART 1: MLFLOW EXPERIMENT TRACKING
# ==========================================
mlflow.set_experiment("Production_Churn_Experiment")

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForest_Baseline"):
    n_estimators = 100
    max_depth = 8
    
    # 1. Log Hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # 2. Train Model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Evaluate Metrics
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # 4. Log Evaluation Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # 5. Log Trained Model Artifact
    mlflow.sklearn.log_model(clf, "model")
    
    print(f"MLflow Run Recorded | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ==========================================
# PART 2: FASTAPI PRODUCTION MICROSERVICE
# ==========================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Pydantic Schema Input Validation
class PredictionInput(BaseModel):
    feature_vector: list[float] = Field(..., example=[0.5, -1.2, 0.8, 0.1, 2.0, -0.4, 0.9, -0.1, 0.3, 0.7])

class PredictionOutput(BaseModel):
    status: str
    prediction: int
    probability: float

app = FastAPI(title="ML Model Inference Service")

@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(payload: PredictionInput):
    try:
        if len(payload.feature_vector) != 10:
            raise HTTPException(status_code=400, detail="Feature vector length must be exactly 10")
            
        x_in = [payload.feature_vector]
        pred = int(clf.predict(x_in)[0])
        prob = float(clf.predict_proba(x_in)[0][pred])
        
        return PredictionOutput(
            status="success",
            prediction=pred,
            probability=round(prob, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 4. Real-World Production Gotchas & Failure Modes

### Data Drift vs Concept Drift
- **Data Drift**: Incoming input feature distributions change over time (e.g., user age demographic shifts).
- **Concept Drift**: The mathematical relationship between input $X$ and target $y$ changes (e.g., consumer purchasing patterns change after economic shifts).
- *Fix*: Implement automated drift monitoring checks using `Evidently AI` or `Great Expectations`.

---

## 5. Feynman Exercises & Deep Thinking Challenges

1. **Question**: Why is log-tracking code versions (Git commit hashes) alongside model artifacts critical in regulated production systems?
   - *Answer*: Guarantees auditability and exact reproducibility if a model prediction must be explained legally months later.

2. **Exercise**: Launch the FastAPI server locally using `uvicorn main:app --reload` and query the `/predict` endpoint via cURL.
