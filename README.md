# SpecWorth — Explainable Laptop Valuation & Uncertainty Estimation

> An end-to-end machine learning system that estimates fair laptop prices from hardware specifications, quantifies prediction uncertainty, explains individual predictions, and identifies potentially underpriced or overpriced listings.

---
Live Link :- https://specworth-explainable-laptop-valuation-uncertainty-estimation.streamlit.app/

## 📌 Overview

**SpecWorth** is an end-to-end machine learning project for **laptop fair-price estimation** using real-world Indian-market laptop listings.

Instead of treating laptop price prediction as a simple regression problem, SpecWorth builds a complete ML workflow:

- Cleans and standardizes raw laptop specifications
- Extracts structured information from unstructured CPU, processor, and GPU text
- Engineers domain-specific hardware features
- Compares multiple regression algorithms
- Performs automated preprocessing and feature selection
- Tunes an XGBoost regression model
- Evaluates performance against a naive baseline
- Analyzes errors across different price segments
- Provides model-level and local prediction explanations
- Uses **split conformal prediction** to estimate an approximately 80% prediction interval
- Deploys the complete system through an interactive Streamlit application
- Provides a deal-checking workflow by comparing listed price against estimated fair value

The project is designed as a **complete ML application**, not just a notebook-based prediction experiment.

---

## 🎯 Problem Statement

Laptop specifications are often presented as a mixture of structured and unstructured information:

- CPU descriptions
- Processor generations and families
- GPU names
- RAM configurations
- Storage specifications
- Display resolutions
- Operating systems
- Brand information

Two laptops with similar specifications can also have substantially different market prices because of differences in processing capability, graphics hardware, memory, storage, display characteristics, and brand.

The goal of SpecWorth is therefore to answer:

> **Given a laptop's specifications, what is a reasonable market price for it, and how uncertain is that estimate?**

The system additionally answers:

> **Is a real laptop listing priced below, near, or above its estimated fair value?**

---

## ✨ Key Features

### 1. Domain-Specific Feature Engineering

Raw specification strings are transformed into meaningful numerical and categorical features.

#### CPU Features

Extracted from CPU descriptions:

- Total cores
- Threads
- Performance cores
- Efficiency cores
- Hybrid CPU indicator
- Core type
- Hyperthreading
- Threads per core

#### Processor Features

Extracted from processor descriptions:

- Processor brand
- Processor family
- Processor generation

#### GPU Features

Extracted from GPU descriptions:

- GPU brand
- GPU family
- VRAM
- Integrated GPU indicator
- Dedicated GPU indicator

#### Display Features

Resolution and screen size are transformed into:

- **Pixels Per Inch (PPI)**

#### Other Features

- RAM capacity
- RAM type
- ROM capacity
- SSD indicator
- Warranty
- Operating-system family

---

## 🧠 Machine Learning Pipeline

```text
Raw Laptop Listings
        │
        ▼
Data Cleaning & Standardization
        │
        ▼
Domain-Specific Feature Engineering
        │
        ├── CPU Parsing
        ├── Processor Parsing
        ├── GPU Parsing
        ├── PPI Calculation
        ├── Storage Processing
        └── OS Classification
        │
        ▼
Train / Test Split
        │
        ▼
Log Transformation of Target
        │
        ▼
ColumnTransformer
        │
        ├── Numerical → Imputation → Scaling
        ├── Categorical → Imputation → One-Hot Encoding
        └── Binary → Passthrough
        │
        ▼
Cross-Validated Model Comparison  (5-fold)
        │
        │   every step below is refit inside each fold,
        │   so nothing is learned from the validation half
        │
        ├── ColumnTransformer
        ├── Random Forest Feature Selection
        └── Estimator
              ├── Linear Regression
              ├── Decision Tree
              ├── Random Forest
              └── XGBoost
        │
        ▼
XGBoost Hyperparameter Tuning  (GridSearchCV over the same pipeline)
        │
        ▼
Fit / Calibration Split
        │
        ├── Final model fit on the fit half
        └── Conformal quantile from the calibration half
        │
        ▼
Point Estimate + 80% Interval  (same model)
        │
        ├── Baseline Comparison
        ├── Global Metrics
        ├── Segment Analysis
        ├── Residual Analysis
        └── Feature Importance
        │
        ▼
Split Conformal Calibration
        │
        ▼
Price + Prediction Interval
        │
        ▼
Streamlit Application
```

---

## 📊 Model Performance

The deployed model was evaluated on the untouched test set. Note the data budget: the
model is fit on 571 listings, 143 are reserved to calibrate the conformal
quantile, and 179 are held out for this evaluation.

| Metric | Result |
|---|---:|
| **R²** | **0.9264** |
| **MAE** | **₹9,842.09** |
| **RMSE** | **₹15,883.78** |

A naive baseline was also implemented by predicting the **mean training-set price for every test observation**, providing a reference point against which the final XGBoost model can be evaluated.

---

## 📈 Model Comparison

Four regression models were evaluated using 5-fold cross-validation on the log-transformed target.

| Model | CV R² | CV MAE (log) | CV RMSE (log) |
|---|---:|---:|---:|
| **XGBoost** | **0.8771** | **0.1419** | **0.2106** |
| Random Forest | 0.8711 | 0.1449 | 0.216 |
| Linear Regression | 0.855 | 0.1566 | 0.2284 |
| Decision Tree | 0.7886 | 0.1765 | 0.2753 |

Every step of the pipeline — imputation, scaling, one-hot encoding and feature selection — is refit inside each fold, so nothing is learned from the validation half.

XGBoost achieved the strongest cross-validation R² among the evaluated models and was subsequently selected for hyperparameter tuning.

---

## ⚙️ Hyperparameter Optimization

The XGBoost model was tuned using `GridSearchCV`.

The selected configuration was:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `max_depth` | 4 |
| `min_child_weight` | 3 |

The final tuned model was then evaluated on the untouched test set.

---

## 🎯 Target Transformation

Laptop prices exhibit a right-skewed distribution.

To reduce the influence of extreme prices and provide a more stable regression target, the project uses:

```python
y_log = np.log1p(y)
```

Predictions are converted back to the original price scale using:

```python
np.expm1(prediction)
```

Therefore, model training occurs in log-price space while final business-facing metrics and predictions are reported in Indian Rupees.

---

## 📉 Segment-Wise Error Analysis

| Segment | n | MAE | R² | MAPE |
|---|---:|---:|---:|---:|
| Budget | 63 | ₹5,087.83 | 0.355 | 14.17% |
| Mid-range | 65 | ₹6,618.87 | -0.061 | 10.58% |
| Premium | 51 | ₹19,823.01 | 0.853 | 13.51% |

R² is **not comparable across these segments**, and the negative mid-range value is
structural rather than a sign of failure. Segments are price terciles, so the middle
band has the narrowest range of actual prices and therefore the smallest total sum of
squares. Dividing a similar squared error by a much smaller denominator drives R²
negative by construction. Sample size is not the cause — the mid-range segment has the
*most* test observations of the three.

MAPE is the metric to compare across segments, and it is roughly flat, which is the
honest reading: relative accuracy is similar everywhere, while absolute error scales
with price level.
---

## 📏 Uncertainty Estimation with Conformal Prediction

A point prediction alone does not communicate how uncertain the estimate is.

SpecWorth therefore uses **split conformal prediction** to construct an approximately 80% prediction interval around the estimated price.

The point estimate and the interval bounds are produced by the **same** model — the one whose calibration residuals set the conformal quantile. This is what makes the reported coverage figure meaningful: centring the interval on a different, better-fit model would void the guarantee.

### Workflow

```text
Training Data
     │
     ├───────────────┐
     ▼               ▼
Model-Fitting     Calibration
     │               │
     │          Calibration Errors
     │               │
     └───────┬───────┘
             ▼
       Conformal Quantile
             │
             ▼
      Held-Out Test Set
             │
             ▼
   Lower ─ Point ─ Upper
```

The resulting conformal interval achieved:

> **86.6% empirical coverage on the 179-listing held-out test set** (target 80%)

Two caveats worth stating plainly. First, the finite-sample rule selects the
⌈(n+1)(1−α)⌉-th of 143 sorted calibration residuals, so the quantile actually
targets roughly 80.6% rather than exactly 80% — mild over-coverage is built in.
Second, with only 143 calibration points the quantile is itself a noisy estimate, and
this particular fit/calibration split landed on the conservative side. The intervals are
therefore somewhat wider than strictly necessary. Over-coverage is the safe direction, but
it should not be read as evidence of a well-centred interval; averaging coverage over
repeated random splits would give a more defensible figure.

The interval is designed to communicate uncertainty rather than simply displaying a fixed percentage margin around the predicted price.

---

## 🔍 Model Explainability

SpecWorth provides two levels of interpretability.

### Global Feature Importance

XGBoost's built-in feature importance is used to identify features that contributed most strongly to the final model.

The most influential features include characteristics related to:

- CPU core count
- RAM
- CPU threads
- GPU VRAM
- CPU architecture
- Storage
- GPU brand
- Processor family
- Hyperthreading
- Display characteristics

### Local Prediction Explanation

For an individual laptop, SpecWorth uses a **perturbation-based local explanation**.

The process is:

```text
Original Laptop
      │
      ▼
Base Prediction
      │
      ▼
Replace One Feature
with a Typical Training Value
      │
      ▼
Predict Again
      │
      ▼
Prediction Difference
      │
      ▼
Feature Contribution
```

This allows the application to communicate which characteristics are pushing a particular prediction higher or lower relative to a typical laptop.

---

## 💰 Deal Detection

SpecWorth also evaluates real listings against the model's estimated fair price.

The system calculates:

```text
Price Difference %
=
(Listed Price - Predicted Price)
/
Predicted Price × 100
```

The result is classified into categories such as:

| Difference | Interpretation |
|---:|---|
| ≤ -10% | Good deal — priced well below estimate |
| -10% to -3% | Slightly below estimate |
| -3% to +3% | Fairly priced |
| +3% to +10% | Slightly above estimate |
| ≥ +10% | Potentially overpriced |

This turns the model from a simple regression system into a practical **price decision-support tool**.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application with three workflows.

### 🗂️ Quick Pick

Select an existing laptop from the dataset and view:

- Estimated fair price
- Prediction interval
- Local feature explanation
- Listing information

### 🛠️ Custom Build

Enter laptop specifications manually, including:

- Brand
- Processor
- CPU configuration
- RAM
- RAM type
- Storage
- Storage type
- GPU
- Display size
- Resolution
- Operating system
- Warranty

The application transforms these raw inputs using the **same feature-engineering functions used during training** before passing them to the model.

### 🔍 Deal Checker

Enter a real laptop listing and its listed price to determine whether it appears:

- Underpriced
- Fairly priced
- Slightly overpriced
- Significantly overpriced

---

## 🔄 Training–Inference Consistency

A key design decision is keeping feature-engineering logic centralized in:

```text
feature_engineering.py
```

The same functions are used by both:

```text
Training Notebook
       │
       └── feature_engineering.py

Streamlit Application
       │
       └── feature_engineering.py
```

This prevents **training-serving feature drift**, where the features generated during application inference differ from those used during model training.

---

## 📁 Project Structure

```text
SpecWorth/
│
├── app.py
├── utils.py
├── feature_engineering.py
├── style.css
├── requirements.txt
├── README.md
│
├── main.ipynb
│
├── data/
│   └── laptops_raw.csv
│
└── models/
    ├── price_model.pkl
    ├── conformal_quantile.pkl
    ├── model_input_columns.pkl
    ├── lookup_df.pkl
    ├── dropdowns.pkl
    ├── feature_reference.pkl
    └── metrics.json
```

### File Responsibilities

| File / Directory | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `utils.py` | Application-side feature construction, prediction, explanation, and deal scoring |
| `feature_engineering.py` | Shared CPU, processor, GPU, PPI, and OS feature extraction |
| `requirements.txt` | Python dependencies |
| `main.ipynb` | Complete model training, calibration, and evaluation pipeline |
| `data/` | Raw laptop dataset |
| `price_model.pkl` | Deployed model — produces both the point estimate and the interval |
| `conformal_quantile.pkl` | Conformal calibration value |
| `model_input_columns.pkl` | Expected model input schema |
| `lookup_df.pkl` | Cleaned listing catalog used by the application |
| `dropdowns.pkl` | Application dropdown options |
| `feature_reference.pkl` | Training-derived reference values used for local explanations |
| `metrics.json` | Stored model evaluation and analysis results |

---

## 🗃️ Dataset

**Source:** [Kaggle — Laptop Price Prediction Dataset](https://www.kaggle.com/datasets/jacksondivakarr/laptop-price-prediction-dataset)

The project uses:

- **893 Indian-market laptop listings**
- Laptop brand
- Product name
- Price
- Processor
- CPU configuration
- RAM
- RAM type
- ROM
- ROM type
- GPU
- Display size
- Resolution
- Operating system
- Warranty
- Specification rating

The raw dataset is cleaned and transformed before modeling.

---

## 🛠️ Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Model Persistence

- Joblib
- JSON

### Deployment

- Streamlit

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SpecWorth
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app.py
```

The Streamlit application will launch in your browser.

---

## 🧪 Reproducing the Training Pipeline

The notebook:

```text
main.ipynb
```

contains the complete training workflow.

Run the notebook from top to bottom to:

1. Load the raw dataset
2. Clean and standardize columns
3. Explore the data
4. Engineer domain-specific features
5. Create the modeling dataset
6. Split training and test data
7. Apply log transformation to the target
8. Build the preprocessing pipeline
9. Compare regression models
10. Tune XGBoost
11. Evaluate the final model
12. Compare against the naive baseline
13. Perform segment-wise error analysis
14. Perform residual analysis
15. Generate feature importance
16. Calibrate conformal prediction intervals
17. Save all artifacts required by the application

---

## 📌 Key Design Decisions

### Why XGBoost?

The dataset contains a mixture of numerical, categorical, and engineered features with potentially nonlinear relationships.

XGBoost performed best among the evaluated candidate models during cross-validation and was therefore selected for final tuning.

### Why log-transform the target?

Laptop prices are right-skewed. Applying `log1p` reduces the influence of extreme prices and provides a more stable regression target.

### Why feature selection?

The preprocessing stage creates additional features through one-hot encoding. Random Forest-based feature selection reduces the resulting feature space before the final XGBoost model.

### Why conformal prediction?

A point estimate alone does not communicate uncertainty. Conformal calibration provides a principled way to construct prediction intervals with a target coverage level.

### Why segment-wise evaluation?

A strong global metric can hide poor performance in specific price ranges. Segment-wise analysis provides a more detailed view of where the model performs well or struggles.

---

## ⚠️ Limitations

Despite strong test-set performance, the system has several limitations:

- The dataset contains only 893 listings.
- Laptop prices can change over time due to discounts, new hardware releases, inventory, and market conditions.
- The model learns relationships present in the dataset and may not generalize equally well to future market conditions.
- Brand effects may reflect market positioning present in the source data rather than intrinsic hardware value.
- The prediction interval provides uncertainty around the model's estimate; it does not guarantee that a listing will actually sell at that price.
- Segment-level R² is not comparable across segments: the segments are price terciles, so the middle band has the narrowest price range and therefore the smallest total sum of squares. MAPE is the metric to read across segments.
- The conformal quantile is calibrated on a small held-out set, so it is itself a noisy estimate; a few points of deviation from the target coverage is expected.

Therefore, SpecWorth should be interpreted as a **data-driven valuation and decision-support system**, not a guaranteed market-price oracle.

---

## 🚀 Future Improvements

Potential extensions include:

- Larger and more recent laptop datasets
- Automated market-price data collection
- Time-aware validation for changing laptop prices
- SHAP-based interactive explanations
- More robust conformal calibration strategies
- Model monitoring and drift detection
- Automated retraining as new listings become available
- Cloud deployment
- API-based inference
- Price tracking over time

---

## 📜 License

This project is intended for educational, portfolio, and demonstration purposes.

The underlying dataset is sourced from Kaggle and remains subject to its original licensing and usage terms.

---

## 👤 Author

**Avishkar Jadhav**

Machine Learning • Data Science • Mathematical Computing

---

## ⭐ Project Summary

```text
                    SPECWORTH
                       │
                       ▼
              Laptop Specifications
                       │
                       ▼
             Domain Feature Engineering
                       │
                       ▼
              ML Pipeline & XGBoost
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Fair Price    Explanation   Uncertainty
          │            │            │
          │            │       Conformal
          │            │       Prediction
          └────────────┼────────────┘
                       ▼
                 Deal Detection
                       │
                       ▼
                Streamlit App
```

**SpecWorth transforms laptop specifications into an explainable, uncertainty-aware estimate of fair market value.**
