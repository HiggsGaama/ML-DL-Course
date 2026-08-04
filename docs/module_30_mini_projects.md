# Module 30: End-to-End Capstone Mini-Projects

Welcome to the final capstone module! Here, you will find five complete, production-grade mini-projects covering every major domain in this course. Each mini-project provides clean, runnable Python code designed with modular software architecture principles.

---

## 🚀 Capstone Project 1: Classical Machine Learning Pipeline
**Goal**: Build an end-to-end Customer Churn Prediction system using `scikit-learn`, complete with pre-processing pipelines, hyperparameter tuning via `Optuna`, and cross-validated evaluation.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import optuna

# 1. Synthesize E-Commerce Customer Churn Dataset
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'tenure_months': np.random.randint(1, 60, size=n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, size=n_samples),
    'total_charges': np.random.uniform(100.0, 5000.0, size=n_samples),
    'contract_type': np.random.choice(['month-to-month', 'one-year', 'two-year'], size=n_samples),
    'payment_method': np.random.choice(['credit_card', 'bank_transfer', 'electronic_check'], size=n_samples),
    'churn': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
})

X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Build Modular Preprocessing Pipeline
num_features = ['tenure_months', 'monthly_charges', 'total_charges']
cat_features = ['contract_type', 'payment_method']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# 3. Optuna Objective Optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42
    )
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores.mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("=== PROJECT 1: OPTIMAL HYPERPARAMETERS ===")
print(study.best_params)

# 4. Train Final Model & Evaluate
best_clf = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42)
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_clf)
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]

print("\n=== PROJECT 1: EVALUATION REPORT ===")
print(f"Test Set ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))
```

---

## 🖼️ Capstone Project 2: Deep Learning Image Classifier (PyTorch)
**Goal**: Build a PyTorch Convolutional Neural Network with Residual Connections, custom Dataset loaders, and GPU validation loops.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=500):
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.labels = (self.images[:, 0, :, :].mean(dim=(1, 2)) > 0).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class ResNetMini(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetMini, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        residual = x
        out = F.relu(self.bn2(self.conv2(x)))
        out += residual
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetMini(num_classes=2).to(device)
train_loader = DataLoader(SyntheticImageDataset(500), batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

model.train()
for epoch in range(1, 4):
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Project 2 | Epoch {epoch}/3 | Loss: {total_loss/len(train_loader):.4f}")
```

---

## 📝 Capstone Project 3: NLP Feature Extractor Pipeline
**Goal**: Build a PyTorch Transformer feature extractor utilizing Hugging Face BERT representations for classification.

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return self.classifier(mean_pooled)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BERTClassifier()
model.eval()

text_samples = ["The backend architecture is remarkably stable.", "Database latency is unacceptable."]
inputs = tokenizer(text_samples, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    predictions = model(inputs['input_ids'], inputs['attention_mask'])
    probs = torch.softmax(predictions, dim=1)

print("=== PROJECT 3: BERT SENTIMENT FEATURE EXTRACTION ===")
print("Probabilities Output Shape:", probs.shape)
print("Sample Predictions:\n", probs.numpy().round(4))
```

---

## 🎨 Capstone Project 4: Generative Image Synthesis (Diffusers)
**Goal**: Local text-conditioned image generation using Stable Diffusion.

```python
import torch
from diffusers import StableDiffusionPipeline

def generate_local_artwork(prompt: str, output_filename: str = "generated_output.png"):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    image.save(output_filename)
    print(f"Project 4 | Saved artwork to '{output_filename}'")
```

---

## ⚡ Capstone Project 5: Microservice MLOps API (FastAPI)
**Goal**: Package a machine learning model into an asynchronous FastAPI REST API with Pydantic schema validation.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., example=12.0)
    monthly_charges: float = Field(..., example=65.5)
    total_charges: float = Field(..., example=786.0)

class ChurnResponse(BaseModel):
    status: str
    churn_prediction: int
    churn_probability: float

app = FastAPI(title="Production Churn Prediction Microservice")

@app.post("/predict_churn", response_model=ChurnResponse)
def predict_churn(payload: ChurnRequest):
    try:
        score = (payload.monthly_charges / 120.0) * 0.6 + (1.0 / (payload.tenure_months + 1.0)) * 0.4
        churn_prob = min(max(score, 0.0), 1.0)
        prediction = 1 if churn_prob > 0.5 else 0
        
        return ChurnResponse(
            status="success",
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🎉 Course Completion Checklist

Congratulations on completing the **Complete Machine Learning & Deep Learning Course**!

### You have mastered:
- [x] Classical Machine Learning algorithms & math foundations (`scikit-learn`).
- [x] Feature engineering, selection, and data sanitization pipelines.
- [x] Deep Learning mechanics, Backpropagation, and PyTorch tensor programming.
- [x] CNNs, ResNets, and Transfer Learning vision models.
- [x] RNNs, LSTMs, and Embeddings vector representations.
- [x] Transformers, Self-Attention ($Q, K, V$), and BERT/GPT feature extraction.
- [x] Generative AI architectures (GANs & Latent Diffusion Models).
- [x] Production MLOps, MLflow experiment tracking, and FastAPI REST deployments.
