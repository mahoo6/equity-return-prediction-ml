# linear_models.py
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LinearModels:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.dates = {}
        self.y_data = {}
    
    def fit_all_models(self, date_train, date_val, date_test, X_train, X_val, X_test, y_train, y_val, y_test):
        
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        # Save for plots 
        self.dates = {
            'train': X_train.index.unique(),
            'val': X_val.index.unique(), 
            'test': X_test.index.unique()
        }
        self.y_data = {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
        
        # Standardization 
        X_train_val_scaled = self.scaler.fit_transform(X_train_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # OLS
        print("-- OLS ---")
        # with sklearn 
        ols_sklearn = LinearRegression()
        ols_sklearn.fit(X_train_val_scaled, y_train_val.values.ravel())
        y_pred_ols = ols_sklearn.predict(X_test_scaled)
        y_pred_train_ols = ols_sklearn.predict(X_train_val_scaled)  
        
        # with statsmodels 
        X_const = sm.add_constant(X_train_val_scaled)
        ols_sm = sm.OLS(y_train_val.values.ravel(), X_const).fit()

        y_pred_test = y_pred_ols.flatten()
        y_test_flat = y_test.values.flatten()
        

        df_pred = pd.DataFrame(
            {'predicted': y_pred_test,
            'actual_return': y_test_flat
            }, index = y_test.index)
        pred_by_month = df_pred.groupby('date')

        ic_series = pred_by_month.apply(
            lambda x: pd.Series({
                'ic': spearmanr(x['predicted'], x['actual_return'])[0],
                'rmse': np.sqrt(mean_squared_error(x['predicted'], x['actual_return']))
            })
        )


        results['OLS'] = {
            'model_sklearn': ols_sklearn,
            'model_statsmodels': ols_sm,
            'predictions': y_pred_ols,
            'predictions_train': y_pred_train_ols,
            'mse': mean_squared_error(y_test.values.ravel(), y_pred_ols),
            'r2': r2_score(y_test.values.ravel(), y_pred_ols),
            'ic': ic_series,
            'sharpe_ratio': sharpe_ratio(y_test.values.ravel()),
            'summary': ols_sm.summary()
        }
        self.print_ols_summary()

        plot_only_predictions(date_test, y_pred_ols, y_test, title='OLS Model Predictions on Test Set', filename='ols_predictions.png')
        plot_ic_distribution(ic_series, filename='ols_ic_distribution.png')        
        plot_rolling_ic(ic_series, filename='ols_rolling_ic.png')
        
        
        # Ridge CV
        print("--- Ridge With Cross Validation ---")
        tscv = TimeSeriesSplit(n_splits=5)
        alphas = np.logspace(-4, 3, 50)
        
        ridge = RidgeCV(alphas=alphas, cv=tscv)
        ridge.fit(X_train_val_scaled, y_train_val.values.ravel())
        y_pred_ridge = ridge.predict(X_test_scaled)
        y_pred_train_ridge = ridge.predict(X_train_val_scaled) 
        y_pred_test = y_pred_ridge.flatten()

        df_pred = pd.DataFrame(
            {'predicted': y_pred_test,
            'actual_return': y_test_flat
            }, index = y_test.index)
        pred_by_month = df_pred.groupby('date')
        
        ic_series = pred_by_month.apply(
            lambda x: pd.Series({
                'ic': spearmanr(x['predicted'], x['actual_return'])[0],
                'rmse': np.sqrt(mean_squared_error(x['predicted'], x['actual_return']))
            })
        )

        results['Ridge'] = {
            'model': ridge,
            'predictions': y_pred_ridge,
            'predictions_train': y_pred_train_ridge,
            'mse': mean_squared_error(y_test.values.ravel(), y_pred_ridge),
            'r2': r2_score(y_test.values.ravel(), y_pred_ridge),
            'best_alpha': ridge.alpha_
        }
        plot_only_predictions(date_test=date_test, y_pred=y_pred_ridge, y_test=y_test, title='RidgeCV Model Predictions on Test Set', filename='ridge_predictions.png')
        plot_ic_distribution(ic_series, filename='ridge_ic_distribution.png')
        plot_rolling_ic(ic_series, filename='ridge_rolling_ic.png')

        # Lasso CV
        print("--- Lasso with Cross Validation ---")
        lasso = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
        lasso.fit(X_train_val_scaled, y_train_val.values.ravel())
        y_pred_lasso = lasso.predict(X_test_scaled)
        y_pred_train_lasso = lasso.predict(X_train_val_scaled) 
        y_pred_test = y_pred_lasso.flatten()


        df_pred = pd.DataFrame(
            {'predicted': y_pred_test,
            'actual_return': y_test_flat
            }, index = y_test.index)
        
        pred_by_month = df_pred.groupby('date')
        
        ic_series = pred_by_month.apply(
            lambda x: pd.Series({
                'ic': spearmanr(x['predicted'], x['actual_return'])[0],
                'rmse': np.sqrt(mean_squared_error(x['predicted'], x['actual_return']))
            })
        )
        
        results['Lasso'] = {
            'model': lasso,
            'predictions': y_pred_lasso,
            'predictions_train': y_pred_train_lasso,
            'mse': mean_squared_error(y_test.values.ravel(), y_pred_lasso),
            'r2': r2_score(y_test.values.ravel(), y_pred_lasso),
            'best_alpha': lasso.alpha_,
            'n_features_selected': np.sum(lasso.coef_ != 0)
        }
        plot_only_predictions(date_test, y_pred_lasso, y_test, title='LassoCV Model Predictions on Test Set', filename='lasso_predictions.png')
        plot_ic_distribution(ic_series, filename='lasso_ic_distribution.png')        
        plot_rolling_ic(ic_series, filename='lasso_rolling_ic.png')


        self.models = results
        return results
    



    def print_results(self):
        print("\n--- Results ---")
        
        for name, result in self.models.items():
            print(f"\n {name}:")
            print(f"   MSE: {result['mse']:.6f}")
            print(f"   R²:  {result['r2']:.4f}")
            
            if 'best_alpha' in result:
                print(f"Best Alpha : {result['best_alpha']:.6f}")
            
            if 'n_features_selected' in result:
                print(f" Number of Selected Features : {result['n_features_selected']}")
    
    def print_ols_summary(self):
        """Affiche le summary complet de l'OLS"""
        if 'OLS' in self.models:
            print("\n---OLS Summary---")
            print("=" * 60)
            print(self.models['OLS']['summary'])



def plot_only_predictions(date_test, y_pred, y_test, title='Model Predictions', filename='predictions.png'):
    """
    Print the actual and predicted values.
    """
    y_pred = pd.Series(y_pred, index=y_test.index)

    y_pred = y_pred.groupby(y_pred.index).mean().copy().cumsum()
    y_test = y_test.groupby(y_test.index).mean().copy().cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(date_test, y_test, label='Test', color='green')
    plt.plot(date_test, y_pred, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()
        
'''
def plot_only_training(date_train, y_pred_train,y_train,  title='Training Data'):

    y_pred_train = pd.Series(y_pred_train)  
    y_pred_train.index = y_train.index


    y_pred_train = y_pred_train.groupby(y_pred_train.index).mean().copy().cumsum()

    y_train = y_train.groupby(y_train.index).mean().copy().cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(date_train, y_train, label='Train', color='blue')
    plt.plot(date_train, y_pred_train, label='Predicted Train', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.show(block=False)

'''

def plot_ic_distribution(df, ax=None, filename='ic_distribution.png'):
    if ax is not None:
        sns.distplot(df.ic, ax=ax)
    else:
        ax = sns.distplot(df.ic)
    mean, median = df.ic.mean(), df.ic.median()
    ax.axvline(0, lw=1, ls='--', c='k')
    ax.text(x=.05, y=.9,
            s=f'Mean: {mean:8.2f}\nMedian: {median:5.2f}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Information Coefficient')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()

def plot_rolling_ic(df, filename='rolling_ic.png'):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(14, 8))
    rolling_result = df.sort_index().rolling(21).mean().dropna()
    mean_ic = df.ic.mean()
    rolling_result.ic.plot(ax=axes[0],
                           title=f'Information Coefficient (Mean: {mean_ic:.2f})',
                           lw=1)
    axes[0].axhline(0, lw=.5, ls='-', color='k')
    axes[0].axhline(mean_ic, lw=1, ls='--', color='k')

    mean_rmse = df.rmse.mean()
    rolling_result.rmse.plot(ax=axes[1],
                             title=f'Root Mean Squared Error (Mean: {mean_rmse:.2%})',
                             lw=1,
                             ylim=(0, df.rmse.max()))
    axes[1].axhline(df.rmse.mean(), lw=1, ls='--', color='k')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.close()



def sharpe_ratio(returns):
  return np.round(np.sqrt(12) * returns.mean() / (returns.std() + 1e-8), 2)