# Machine Learning & Deep Learning: Zero-to-Hero for Software Engineers

Welcome to the **Complete ML & Deep Learning Course**—a battle-tested, developer-first curriculum designed to transition a software engineer into an expert-level Machine Learning and Deep Learning practitioner.

---

## 🎯 Target Audience & Philosophy

This course is built for software engineers who:
- Think in algorithms, data structures, APIs, and design patterns.
- Want to understand **how and why** algorithms work under the hood (no magic, no hand-waving).
- Prefer **intuitive plain-English math explanations first** before looking at formal notation.
- Demand production-ready code examples (`scikit-learn`, `PyTorch`, `MLflow`, `FastAPI`).

---

## 🚀 Recommended Learning Path & Setup

### Prerequisites
- Python 3.10+
- Basic knowledge of arrays/matrices and functions.

### Quickstart Environment Setup
```bash
# Create and activate virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install core scientific and ML/DL stack
pip install numpy pandas scipy matplotlib seaborn scikit-learn optuna xgboost lightgbm torch torchvision torchaudio transformers diffusers mlflow fastapi uvicorn
```

---

## 📚 Complete Syllabus Map

### Part 1: Classical Machine Learning

| Module | Title | Key Topics | Link |
|---|---|---|---|
| **01** | What is Machine Learning? | Types (Supervised, Unsupervised, RL), ML Workflow, Train/Test Split, Overfitting/Underfitting | [Module 01](course/module_01_what_is_ml) |
| **02** | Data Preprocessing | Cleaning, Imputation, Encoding (One-Hot, Target), Scaling (Standard, Robust) | [Module 02](course/module_02_data_preprocessing.md) |
| **03** | Linear & Logistic Regression | Hypothesis, MSE & Log-Loss, Gradient Descent, Sigmoid & Decision Boundaries | [Module 03](course/module_03_linear_logistic_regression) |
| **04** | KNN, Naive Bayes & SVM | Distance metrics, Bayes Theorem, Support Vectors & Kernel Trick | [Module 04](course/module_04_knn_naive_bayes_svm.md) |
| **05** | Decision Trees & Boosting | Tree splits, Entropy/Gini, Random Forests, XGBoost & LightGBM mechanics | [Module 05](course/module_05_tree_models_boosting.md) |
| **06** | Model Evaluation | Confusion Matrix, Precision/Recall, ROC-AUC, Stratified K-Fold CV | [Module 06](course/module_06_model_evaluation.md) |
| **07** | Hyperparameter Tuning | GridSearch, RandomSearch, Optuna Bayesian Optimization | [Module 07](course/module_07_hyperparameter_tuning.md) |
| **08** | Imbalanced Datasets | SMOTE, ADASYN, Class Weights, Focal Loss intuition | [Module 08](course/module_08_imbalanced_datasets.md) |
| **09** | Feature Engineering & Selection | Interaction features, Filter/Wrapper/Embedded methods, RFE, Lasso L1 | [Module 09](course/module_09_feature_engineering_selection.md) |
| **10** | Unsupervised Learning | K-Means, DBSCAN, PCA (Variance & Eigenvectors), t-SNE | [Module 10](course/module_10_unsupervised_learning.md) |
| **11** | Anomaly Detection | Isolation Forests, One-Class SVM, Z-score & Elliptic Envelope | [Module 11](course/module_11_anomaly_detection.md) |
| **12** | Recommendation Systems | Collaborative Filtering (User/Item), Matrix Factorization (SVD), Content-based | [Module 12](course/module_12_recommendation_systems.md) |

---

### Part 2: Deep Learning Essentials

| Module | Title | Key Topics | Link |
|---|---|---|---|
| **13** | Neural Networks Core | Perceptron, Multi-Layer Perceptrons, Activations (ReLU, Softmax), Forward & Backpropagation Intuition | [Module 13](course/module_13_neural_networks.md) |
| **14** | Training Tricks & Optimization | SGD, AdamW, Loss Functions, Dropout, Batch Normalization, Weight Decay | [Module 14](course/module_14_training_tricks.md) |
| **15** | Practical PyTorch | DataLoaders, Custom `nn.Module`, Training Loop, Checkpointing, GPU/CPU Agnostic Code | [Module 15](course/module_15_practical_pytorch.md) |
| **16** | CNNs & Vision Architectures | Convolutions, Kernels, Padding/Stride, Feature Maps, ResNet Residual Blocks | [Module 16](course/module_16_cnns_and_architectures.md) |
| **17** | Transfer Learning | Pre-trained backbones, Feature Extractor vs Fine-Tuning, Differential Learning Rates | [Module 17](course/module_17_transfer_learning.md) |
| **18** | Object Detection | Bounding Boxes, IoU, NMS, Two-Stage (R-CNN) vs One-Stage (YOLO) | [Module 18](course/module_18_object_detection.md) |
| **19** | Autoencoders | Bottleneck, Reconstruction Loss, Variational Autoencoders (VAE), Reparameterization Trick | [Module 19](course/module_19_autoencoders.md) |
| **20** | RNNs & LSTMs | Sequence Modeling, Vanishing Gradients, LSTM Gates (Forget, Input, Output), GRU | [Module 20](course/module_20_rnns_lstms.md) |
| **21** | Embeddings | Word2Vec (Skip-gram, CBOW), FastText, Sentence Embeddings & Vector Similarity | [Module 21](course/module_21_embeddings.md) |
| **22** | Transformers & Self-Attention | Query-Key-Value ($Q, K, V$), Scaled Dot-Product, Multi-Head Attention, Positional Encoding | [Module 22](course/module_22_transformers_attention.md) |
| **23** | BERT & GPT | Encoder vs Decoder, Masked Language Modeling vs Causal LM, Feature Extraction | [Module 23](course/module_23_bert_gpt_feature_extraction.md) |

---

### Part 3: Generative AI & Advanced Models

| Module | Title | Key Topics | Link |
|---|---|---|---|
| **24** | GANs | Minimax Game, Generator vs Discriminator, BCE Loss, Mode Collapse | [Module 24](course/module_24_gans.md) |
| **25** | Diffusion Models | Forward Noise Addition, Reverse Denoising U-Net, DDPM Intuition | [Module 25](course/module_25_diffusion_models.md) |
| **26** | Image Generation | Latent Diffusion Models (Stable Diffusion), Text Conditioning via CLIP, Local Inference | [Module 26](course/module_26_image_generation_sd.md) |
| **27** | Multimodal Models | Contrastive Learning, Joint Embeddings, CLIP, Vision-Language Models Overview | [Module 27](course/module_27_multimodal_models.md) |
| **28** | Real-World AI Pipelines | End-to-end CV Inference Pipelines, NLP Processing Pipelines, Production Patterns | [Module 28](course/module_28_ai_applications.md) |

---

### Part 4: Practical MLOps & Capstones

| Module | Title | Key Topics | Link |
|---|---|---|---|
| **29** | MLOps & Deployment | Experiment Tracking (MLflow), Model Registry, REST API Deployment with FastAPI | [Module 29](course/module_29_mlops_tracking_deployment.md) |
| **30** | Comprehensive Mini-Projects | 5 Practical End-to-End Projects covering ML, Vision, NLP, GenAI, and MLOps | [Module 30](course/module_30_mini_projects.md) |

---

## 🛠️ How to Study Each Module

Every module in this course is formatted consistently:
1. **Developer Mental Model**: Explaining the ML concept using familiar software patterns.
2. **Intuitive Breakdown**: Deep-dive explanation without hand-waving.
3. **Architecture Diagrams**: Visual ASCII / Mermaid representations of models and data flows.
4. **Practical Code**: Production-ready Python snippets (`scikit-learn` or `PyTorch`).
5. **Math & Mechanics**: Plain English intuition first $\rightarrow$ formal equations with LaTeX.
6. **Advanced & Edge Cases**: Optimization tips, trade-offs, and failure modes.
7. **Exercises & Self-Check**: Quick questions to test your understanding.

Start with [Module 01: What is Machine Learning?](course/module_01_what_is_ml.md) and build your way up!
