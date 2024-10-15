import pandas as pd
import torch

def DatasetTimeseries(data, col):
    """
    This is a function to convert to the appropriate timeseries format. I realise this is a oneliner but eh
    
    For now this will a stricly univariate timeseries. I will scale this as needed

    fuck it imma pytorch this bitch

    args:
        data (pd.Dataset): The dataset we need to convert to the appropriate format (this should ideally have a unique row for each frame)
        cols (string): This is a list of strings of the columns we need to preserve (THis should just be one specific column)

    returns:
        ts_data (pd.Dataset): The dataset that is in the format Q ordered data (In our case 240) * timeseries (125)
        results (pd.Series): The results which should be a binary output with 125 observations
    """
    data_output = data[['trial_id', col]].groupby(['trial_id'])[[col]].apply(lambda df: df.reset_index(drop = True)).unstack().reset_index('trial_id')
    data_output.columns = data_output.columns.droplevel(0)
    data_output = data_output.drop('', axis= 1) 

    results = data[['trial_id', 'result']].groupby(['trial_id'])['result'].head(1)
    data_output = torch.tensor(data_output.values)
    labels = torch.tensor(results.map({'missed': 0, 'made':1}).values) 
    return data_output, labels