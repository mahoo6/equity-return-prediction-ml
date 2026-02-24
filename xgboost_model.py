# xgboost_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.top_features = None
        self.feature_importance = None
        self.results = {}

    def fit_and_select_features(self, X_train, X_val, X_test, y_train, y_val, y_test, top_n=100):
        
        
        #We create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train.values, label=y_train.values.ravel())
        dval = xgb.DMatrix(X_val.values, label=y_val.values.ravel())
        dtest = xgb.DMatrix(X_test.values, label=y_test.values.ravel())
        
        # We define the parameters for XGBoost
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # We create the evaluation sets
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        # We train the model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # We compute predictions on the test set
        y_pred = self.model.predict(dtest)
        
        # Feature importance
        importance_dict = self.model.get_score(importance_type='weight')
        feature_names = X_train.columns.tolist()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        }).sort_values('importance', ascending=False)
        
        # Selection of top features
        self.top_features = importance_df.head(top_n)['feature'].tolist()
        self.feature_importance = importance_df
        

        def evaluate(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            strategy_returns = y_pred * y_test.values.ravel()
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(12) #Strategy Sharpe ratio
            return {'R²': r2, 'RMSE': rmse, 'Sharpe_pred': sharpe}
        
        self.results = evaluate(y_test.values.ravel(), y_pred)
        
        print(f"   R²: {self.results['R²']:.4f}")
        print(f"   RMSE: {self.results['RMSE']:.6f}")
        print(f"   Sharpe: {self.results['Sharpe_pred']:.4f}")
        print(f"Top {top_n} features selected")
        
        return self.results
    
    def get_top_features(self):
        if self.top_features is None:
            raise ValueError("No features selected, you should train XGBoost first.")
        return self.top_features
    

    #We filter the datasets based on the selected top features
    def filter_datasets(self, X_train, X_val, X_test):
        if self.top_features is None:
            raise ValueError("No features selected, you should train XGBoost first.")

        print(f"   Before : {X_train.shape[1]} features")
        
        X_train_filtered = X_train[self.top_features]
        X_val_filtered = X_val[self.top_features]
        X_test_filtered = X_test[self.top_features]
        
        print(f"   After : {X_train_filtered.shape[1]} features")
        
        return X_train_filtered, X_val_filtered, X_test_filtered

    #Let's see the feature importance
    def plot_feature_importance(self, top_n=20):
        if self.feature_importance is None:
            print("You should train XGBoost first to get feature importance.")
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances for XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    
    def print_top_features(self, n=20):
        if self.top_features is None:
            print("No features selected")
            return
        
        print(f"Top {n} Features sélectionnées:")
        for i, feature in enumerate(self.top_features[:n], 1):
            importance = self.feature_importance[self.feature_importance['feature'] == feature]['importance'].iloc[0]
            print(f"   {i:2d}. {feature:<40} (importance: {importance:>6.0f})")