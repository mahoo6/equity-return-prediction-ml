# Machine-Learning in Finance

This project aims to forecast monthly U.S. stock returns using machine learning, leveraging financial characteristics observable at the time of prediction. The **target variable** is derived from the **monthly CRSP dataset**, while **predictors** come from **quarterly Compustat fundamentals** and **JKP factor characteristics**, both offering rich insights into firm behavior and asset pricing dynamics.

## 📁 Project Structure

```         
mltambe/
├── main.py                 # Main script to run the full pipeline
├── data_preprocessing.py  # Raw-to-ML dataset construction
├── linear_models.py       # Linear regression and shrinkage models (e.g., Ridge, Lasso)
├── mlp_model.py           # Feedforward neural network (MLP)
├── xgboost_model.py       # Gradient boosting model
├── requirements.txt       # Required dependencies
├── data/
│   ├── Predictors/        # Merged predictors and processed X
│   │   ├── ccmxpf_linktable.csv
│   │   ├── CompFirmCharac.csv
│   │   ├── jkp_characteristic.plk
│   │   ├── merged.plk
│   │   └── X.pkl
│   └── Targets/
│       └── monthly_crsp.csv
│       └── y.pkl
├── plots/                 # Output directory for plots
```

## ✅ Getting Started

### 1. Install Dependencies

Ensure you are using **Python 3.8+** and install required packages:

``` bash
pip install -r requirements.txt
```

> It's recommended to use a virtual environment (`venv` or `conda`).

### 2. Prepare the Data

Download the [raw data](https://drive.google.com/drive/u/1/folders/1R9yTfsztKLJbT5fZxTHNKe1pQFnexrJm?usp=sharing) and place raw data in the appropriate folders:

```         
data/
├── Predictors/
│   ├── ccmxpf_linktable.csv
│   ├── CompFirmCharac.csv
│   ├── jkp_characteristic.plk
└── Targets/
    └── monthly_crsp.csv
```

Processed datasets (`merged.pkl`, `X.pkl`, `y.pkl`) will be automatically generated and saved in the same directories.

### 3. Run the Pipeline

To execute the full workflow from data preprocessing to model training and evaluation:

``` bash
python main.py
```

> This will train models, evaluate them, and save results and plots in the `plots/` directory.

## 🧠 Models Implemented

-   **OLS, Ridge, and Lasso Regression** (`linear_models.py`)
-   **Multi-Layer Perceptron (MLP)** (`mlp_model.py`)
-   **XGBoost Regressor** (`xgboost_model.py`)

## 📊 Outputs

-   Plots of predicted vs. actual returns
-   IC Distribution, Rolling IC
-   All outputs saved in the `plots/` folder
