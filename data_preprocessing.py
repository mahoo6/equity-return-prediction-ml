# data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#To take care of outliers
def winsorize_df(df, columns, lower=0.01, upper=0.99):
    df_winsorized = df.copy()
    for col in columns:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df_winsorized[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df_winsorized

#To clean the datasets 
def clean_data():
    df = pd.read_pickle('data/Predictors/jkp_characteristic.pkl')
    df = df.drop(columns=['Unnamed: 0', 'id','excntry','size_grp'])
    
    chars = pd.read_excel('https://github.com/bkelly-lab/ReplicationCrisis/raw/master/GlobalFactors/Factor%20Details.xlsx')
    chars_names = chars[chars['abr_jkp'].notna()][['name_new', 'abr_jkp']]
    abbrev_to_full = dict(zip(chars_names["abr_jkp"], chars_names["name_new"]))
    df = df.rename(columns=abbrev_to_full)
    df = df.rename(columns={'eom': 'date'})
    
    # We clean the data
    df = df.dropna(axis=1, thresh=0.7*len(df))
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    
    # We use the Linking table from WRDS 
    linking_table = pd.read_csv('data/Predictors/ccmxpf_linktable.csv')
    linking_table = linking_table[['gvkey', 'lpermno', 'linkdt', 'linkenddt']].dropna(subset=['lpermno'])
    linking_table = linking_table.rename(columns={'lpermno': 'permno'})
    linking_table['linkdt'] = pd.to_datetime(linking_table['linkdt'], errors='coerce')
    linking_table['linkenddt'] = pd.to_datetime(linking_table['linkenddt'], errors='coerce')
    linking_table['linkenddt'] = linking_table['linkenddt'].fillna(pd.Timestamp('today'))
    
    # Now we merge with the linking_table on PERMNO
    df = pd.merge(df, linking_table, on='permno', how='left', indicator=True)
    df = df[(df['date'] >= df['linkdt']) & (df['date'] <= df['linkenddt'])]
    df = df.drop(columns=['gvkey_y']).rename(columns={'gvkey_x': 'gvkey'})
    df = df.dropna(subset=['gvkey'])
    df = df.sort_values(by=['gvkey', 'date'])
    df = df.drop_duplicates(subset=['gvkey', 'date'])

    financial_vars = df.select_dtypes(include=[np.number]).drop(columns=['gvkey', 'permno']).columns
    df[financial_vars] = df.groupby('gvkey')[financial_vars].transform(lambda x: x.ffill().bfill())
    df = df.dropna(subset=financial_vars)
    df = df.sort_values(by=['gvkey', 'date'])
    
    # Winsorisation
    df = winsorize_df(df, financial_vars, lower=0.01, upper=0.99)

    # Cleaned data saving
    df.to_pickle('data/Predictors/jkp_characteristic_clean.pkl')
    print(" JKP cleaned and saved")
    
    # Clean of the second dataset: Compustat
    df_comp = pd.read_pickle('data/Predictors/CompFirmCharac.pkl')
    df_comp['datadate'] = pd.to_datetime(df_comp['datadate'], errors='coerce')
    df_comp = df_comp.drop(columns=['fyearq', 'fqtr','fyr','cusip','cik']).rename(columns={'datadate': 'date'})
    df_comp = df_comp.drop(columns=df_comp.select_dtypes(include=['object', 'category']).columns)
    

    df_comp = df_comp.dropna(axis=1, thresh=0.7*len(df_comp))
    financial_vars = df_comp.select_dtypes(include=[np.number]).drop(columns=['gvkey']).columns
    df_comp[financial_vars] = df_comp.groupby('gvkey')[financial_vars].transform(lambda x: x.ffill().bfill())
    df_comp = df_comp.dropna(subset=financial_vars)
    df_comp = df_comp.sort_values(by=['gvkey', 'date'])
    
    # Winsorize
    df_comp = winsorize_df(df_comp, financial_vars, lower=0.01, upper=0.99)
    df_comp = df_comp.sort_values(by=['gvkey', 'date'])
    
    # Sauvegarde
    df_comp.to_pickle('data/Predictors/CompFirmCharac_clean.pkl')
    print("Compustat cleaned and saved")


#To merge the two datasets
def merge_data():
    jkp_df = pd.read_pickle('data/Predictors/jkp_characteristic_clean.pkl')
    compustat_df = pd.read_pickle('data/Predictors/CompFirmCharac_clean.pkl')

    # CRSP : target return 
    crsp = pd.read_csv('data/Targets/monthly_crsp.csv')
    crsp = crsp.drop(columns=['HdrCUSIP', 'CUSIP','Ticker','TradingSymbol','PERMCO', 'SICCD','NAICS','sprtrn'])
    crsp = crsp.rename(columns={'MthCalDt': 'date', 'PERMNO': 'permno', 'MthRet':'Return'})
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp = crsp.ffill(axis=0)
    
    # Merge of JKP + Compustat, the two predictor datasets
    pieces = []
    common_gvkeys = set(jkp_df['gvkey']).intersection(compustat_df['gvkey'])
    
    for gvkey in common_gvkeys:
        jkp_sub = jkp_df[jkp_df['gvkey'] == gvkey].sort_values('date')
        comp_sub = compustat_df[compustat_df['gvkey'] == gvkey].sort_values('date')
        
        if len(comp_sub) == 0:
            continue
            
        merged = pd.merge_asof(jkp_sub, comp_sub, on='date', direction='backward')
        merged['gvkey'] = gvkey
        pieces.append(merged)
    
    final_merged = pd.concat(pieces, axis=0, ignore_index=True).drop(columns=['gvkey_x', 'gvkey_y'])
    
    # Merge with CRSP the target return dataset
    merged = pd.merge(final_merged, crsp, on=['permno', 'date'], how='inner')
    merged = merged.sort_values(by=['gvkey', 'date'])
    merged = merged.set_index('date')
    merged = merged.dropna()
    
    cols = ['gvkey'] + [col for col in merged.columns if col != 'gvkey']
    merged = merged[cols]
    
    merged.to_pickle('data/Predictors/merged_df.pkl')

def shift_and_create_features():

    merged = pd.read_pickle('data/Predictors/merged_df.pkl')

    merged.groupby('date').nunique()['permno'].plot(title='Number of unique permno per date')

    #Filter dates to after 1985, knowing the result of the plot 
    merged = merged[merged.index > pd.to_datetime('1985-01-01')]
    
    cols_to_exclude = ['gvkey', 'cusip', 'Short-term reversal', 'Return']
    cols_to_shift = merged.select_dtypes(include='number').columns.difference(cols_to_exclude)
    
    #Lagging features by one period (t−1) for each firm (grouped by PERMNO)
    merged[cols_to_shift] = merged.groupby('permno')[cols_to_shift].shift(1)
    
    #Delete rows with NaN values after shifting
    merged = merged.dropna(subset=cols_to_shift)
    merged = merged.sort_index()
    
    #Create features and target
    numeric_features = merged.select_dtypes(include='number').columns.difference(cols_to_exclude)
    X = merged[numeric_features]
    y = merged[['Return', 'permno']]
    
    # To save them for later use
    X.to_pickle('data/Predictors/X.pkl')
    y.to_pickle('data/Targets/y.pkl')
    
    return X, y


#Temporal split with time filtering : only companies seen in training remain in val/test
def train_test_split_times_series(X, y, test_size=0.2, val_size=0.1):

    X = X.sort_index()
    y = y.sort_index()

    unique_dates = sorted(X.index.unique())
    q_1 = int(len(unique_dates) * (1 - test_size - val_size))
    q_3 = int(len(unique_dates) * (1 - test_size))

    # Split dates into train, val, and test sets
    date_train = unique_dates[:q_1]
    date_val = unique_dates[q_1:q_3]
    date_test = unique_dates[q_3:]

    mask_train = X.index.isin(date_train)
    mask_val = X.index.isin(date_val)
    mask_test = X.index.isin(date_test)

    # Datasets with firm identifiers before further filtering
    X_train_with_ids = X[mask_train].sort_values(['permno', 'date'])
    X_val_with_ids = X[mask_val].sort_values(['permno', 'date'])
    X_test_with_ids = X[mask_test].sort_values(['permno', 'date'])

    y_train_with_ids = y[mask_train].sort_values(['permno', 'date'])
    y_val_with_ids = y[mask_val].sort_values(['permno', 'date'])
    y_test_with_ids = y[mask_test].sort_values(['permno', 'date'])

    # List of companies in the training set
    train_companies = X_train_with_ids['permno'].unique()

    #Companies exluded from val and test
    excluded_val_companies = set(X_val_with_ids['permno'].unique()) - set(train_companies)
    excluded_test_companies = set(X_test_with_ids['permno'].unique()) - set(train_companies)

    # Filtering
    val_mask = X_val_with_ids['permno'].isin(train_companies)
    test_mask = X_test_with_ids['permno'].isin(train_companies)

    X_val_filtered = X_val_with_ids[val_mask]
    X_test_filtered = X_test_with_ids[test_mask]
    y_val_filtered = y_val_with_ids[val_mask]
    y_test_filtered = y_test_with_ids[test_mask]

    # Drop identification columns
    X_train = X_train_with_ids.drop(columns=['permno'])
    X_val = X_val_filtered.drop(columns=['permno'])
    X_test = X_test_filtered.drop(columns=['permno'])
    y_train = y_train_with_ids.drop(columns=['permno'])
    y_val = y_val_filtered.drop(columns=['permno'])
    y_test = y_test_filtered.drop(columns=['permno'])

    print(f"Unique companies in train: {len(train_companies)}")
    print(f"Rows removed from val: {len(X_val_with_ids) - len(X_val_filtered)}")
    print(f"Rows removed from test: {len(X_test_with_ids) - len(X_test_filtered)}")
    print(f"Companies in val not in train: {len(excluded_val_companies)}")
    print(f"Companies in test not in train: {len(excluded_test_companies)}")
    print(f"Companies remaining in test set: {X_test_filtered['permno'].nunique()}")

    # Plot of average returns over time
    plt.plot(date_train, y_train.groupby(y_train.index).mean(), label='Train')
    plt.plot(date_val, y_val.groupby(y_val.index).mean(), label='Validation')
    plt.plot(date_test, y_test.groupby(y_test.index).mean(), label='Test')
    plt.legend(['Train mean', 'Validation mean', 'Test mean'])
    plt.savefig('plots/train_test_split.png')
    plt.close()

    return date_train, date_val, date_test, X_train, X_val, X_test, y_train, y_val, y_test

