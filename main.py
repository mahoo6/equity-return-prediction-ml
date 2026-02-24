import os
import pandas as pd
from data_preprocessing import clean_data, merge_data,  train_test_split_times_series, shift_and_create_features
from linear_models import LinearModels, plot_only_predictions, plot_ic_distribution, plot_rolling_ic
from mlp_model import MLPModel 
from xgboost_model import XGBoostModel


def main():
    merged_path = 'data/Predictors/merged_df.pkl'

#those "if-else" are for the case where you run the code multiples times.
#the first time you run it : it creates "merged.py","X.py", "y.py", for the others times you only read them so it's way faster

    if os.path.exists(merged_path):
        print("Merged.py found, Checking for X.py and y.py ...")
    else:
        print("Cleaning Data...")
        clean_data()
        
        print("Merging Data...")
        merge_data()
    
    X_path = 'data/Predictors/X.pkl'
    y_path = 'data/Targets/y.pkl'

    if os.path.exists(X_path): #checking only for X because both are created at the same time
        print("X.py and y.py found")
    else:
        print("Creating X (shifted) and y...")
        merged=shift_and_create_features()

    X = pd.read_pickle(X_path)
    y = pd.read_pickle(y_path)

    
    print("Splitting in test and train...")
    date_train, date_val, date_test, X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_times_series(X, y)


    #XGBOOST
    xgb_model = XGBoostModel()
    print("Training with XGBoost...")
    xgb_results = xgb_model.fit_and_select_features(X_train, X_val, X_test, y_train, y_val, y_test, top_n=70)
    
    #Data set only with selected features
    X_train_filtered, X_val_filtered, X_test_filtered = xgb_model.filter_datasets(X_train, X_val, X_test)

    # Linear Models 
    linear_models=LinearModels()
    print("Training of Linear Models...")
    linear_results = linear_models.fit_all_models(date_train, date_val, date_test,X_train_filtered, X_val_filtered, X_test_filtered, y_train, y_val, y_test)
    linear_models.print_results()
    linear_models.print_ols_summary()

    # MLP Model
    top_features = xgb_model.get_top_features()
    #top_features = ['Firm age', 'Price per share', 'Price momentum t-3 to t-1', 'Current price to high price over last year', 'Price momentum t-6 to t-1', 'Market Equity', 'Amihud Measure', 'Price momentum t-9 to t-1', 'Year 1-lagged return, annual', 'Highest 5 days of return', 'Price momentum t-12 to t-7', 'Year 1-lagged return, nonannual', 'oiadpy', 'Return volatility', 'Idiosyncratic volatility from the CAPM (252 days)', 'The high-low bid-ask spread', 'Number of zero trades with turnover as tiebreaker (1 month)', 'Coefficient of variation for share turnover', 'Earnings-to-price', 'Highest 5 days of return scaled by volatility', 'Price momentum t-12 to t-1', 'Ebitda-to-market enterprise value', 'Book-to-market equity', 'Share turnover', 'Book-to-market enterprise value', 'Dimson beta', 'Downside beta', 'exchg', 'Coefficient of variation for dollar trading volume', 'Assets-to-market', 'Maximum daily return', 'Asset Growth', 'Asset tangibility', 'Mispricing factor: Performance', 'Residual momentum t-12 to t-1', 'Coskewness', 'Residual momentum t-6 to t-1', 'Dollar trading volume', 'Idiosyncratic volatility from the CAPM (21 days)', 'Sales-to-market', 'Idiosyncratic volatility from the Fama-French 3-factor model', 'Net debt-to-price', 'Market Beta', 'epspiy', 'epspxy', 'Cash-to-assets', 'Operating profits-to-book assets', 'cshpry', 'Book leverage', 'Number of zero trades with turnover as tiebreaker (6 months)', 'Change in common equity', 'Profit margin', 'Number of zero trades with turnover as tiebreaker (12 months)', 'Net payout yield', 'Gross profits-to-assets', 'Asset turnover', 'Operating cash flow-to-market', 'Quality minus Junk: Profitability', 'CAPEX growth (1 year)', 'Equity net payout', 'Gross profits-to-lagged assets', 'Free cash flow-to-price', 'epsfiy', 'Return on equity', 'Change in current operating liabilities', 'Idiosyncratic skewness from the Fama-French 3-factor model', 'Total skewness', 'Capital turnover', 'Idiosyncratic skewness from the CAPM', 'Quality minus Junk: Safety', 'Sales Growth (1 year)', 'Debt-to-market', 'Net equity issuance', 'Net stock issues', 'epsfxy', 'Change PPE and Inventory', 'Inventory change', 'Change in operating cash flow to assets', 'Return on net operating assets', 'Operating leverage', 'Mispricing factor: Management', 'Change in current operating assets', 'Percent operating accruals', 'Change in noncurrent operating liabilities', 'Net total issuance', 'Cash-based operating profits-to-lagged book assets', 'Cash-based operating profits-to-book assets', 'Net debt issuance', 'Net operating assets', 'Percent total accruals', 'Operating accruals', 'Change in current operating working capital', 'Tax expense surprise', 'Change in financial liabilities', 'Operating profits-to-lagged book assets', 'Operating cash flow to assets', 'Liquidity of book assets', 'Change in net financial assets', 'Total accruals', 'Change in noncurrent operating assets']
    if "permno" in top_features: 
         X_filtered_for_mlp = X[top_features]
    else :
        X_filtered_for_mlp=X[top_features+ ["permno"]]

    #Selecting the 100 companies where we have most data on. (Too long on our computer oif we take more.)
    company_counts = X_filtered_for_mlp['permno'].value_counts()
    top_companies = company_counts.head(100).index.tolist()
    X_short = X_filtered_for_mlp[X_filtered_for_mlp['permno'].isin(top_companies)]
    y_short = y[y['permno'].isin(top_companies)]
    
    print("Doing the train test split again with the small Dataset...")
    date_train, date_val, date_test, X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_times_series(X_short, y_short)

    mlp_model = MLPModel()
    print ("Training MLP...")
    mlp_results = mlp_model.fit_model(X_train, X_val, X_test, y_train, y_val, y_test)
    mlp_model.print_results()
    
    
    print("---FINISHED---")
    return linear_results


if __name__ == "__main__":
    results = main()
