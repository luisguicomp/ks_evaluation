import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Function to plot KS chart
def plot_ks_chart(ks_table):
    """Function to plot KS chart"""
    ks_table['%cum_true_num'] = ks_table['%cum_true'].str.replace('%','').astype('float')
    ks_table['%cum_bad_num'] = ks_table['%cum_bad'].str.replace('%','').astype('float')
    plt.figure(figsize=(10,6))
    plt.plot(ks_table.index, ks_table['%cum_true_num'], 'g')
    plt.plot(ks_table.index, ks_table['%cum_bad_num'], 'r')
    plt.fill_between(ks_table.index, ks_table['%cum_true_num'], ks_table['%cum_bad_num'], color="grey", alpha=0.2, hatch='|')
    plt.xticks(ks_table.index)
    plt.title('KS - Accumulated curves')
    plt.legend(['%true', '%bad'], loc='lower right')
    plt.xlabel('Decile')
    plt.ylabel('% Accumulated')
    plt.show()

def ks_calculate(model, x_val, y_val, target='good', verbose=True, buckets=10):
    """Function to calculate KS metric"""
    pred = model.predict(x_val)
    proba = model.predict_proba(x_val)
    if isinstance(y_val, np.ndarray):
        y_val = pd.DataFrame(y_val, columns=[target])
    df_val_ks = y_val.reset_index(drop=True)
    df2 = pd.DataFrame(proba, columns=['p0','p'])
    data_ks = df2.join(df_val_ks)
    ks_score, kstable = ks_calc(data=data_ks, target=target, prob="p", verbose=verbose, buckets=buckets)
    if verbose == True:
        print("accuracy_score: ", accuracy_score(y_val.values, pred))
        print("roc_auc_score: ", roc_auc_score(y_true= y_val.values, y_score= proba[:,1]))
        print("ks_score: ", ks_score)
    return ks_score, kstable

def ks_calc(data=None, target=None, prob=None, verbose=True, buckets=10):
    """Support function to calculate KS metric"""
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], buckets)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['true']   = grouped.sum()[target]
    kstable['false'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['true_rate'] = (kstable.true / data[target].sum()).apply('{0:.2%}'.format)
    kstable['false_rate'] = (kstable.false / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['%cum_true']=(kstable.true / data[target].sum()).cumsum()
    kstable['%cum_false']=(kstable.false / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['%cum_true']-kstable['%cum_false'], 3) * 100

    #Formating
    kstable['%cum_true']= kstable['%cum_true'].apply('{0:.2%}'.format)
    kstable['%cum_bad']= kstable['%cum_bad'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    if verbose==True:
        print(kstable)
        print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return round(max(kstable['KS']),4), kstable