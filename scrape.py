#%%
import pandas as pd
from fredapi import Fred
from pandas import datetime


fred = Fred(api_key = '13a856ab4d9b2976146eb2e1fddd511d')

df = pd.Series(fred.get_series('CPIAUCSL').dropna(),name='CPI')

df2 = pd.Series(fred.get_series('CBETHUSD',frequency='w').dropna(),name='ETH')

combined = pd.merge(df,df2,how='outer',left_index=True,right_index=True)
combined.index.name='Date'
combined.to_csv('data.csv')
# %%
option_slctd = 'CPI'
data = pd.read_csv('data.csv')
data = data.set_index('Date')
df = data[option_slctd]
df = pd.DataFrame(df).dropna()
df.index = pd.DatetimeIndex(df.index.values)
start_date = datetime(2003,1,1)
end_date = datetime(2021,1,1)
df=df[df.index>start_date]
df=df[df.index<end_date]
# %%
