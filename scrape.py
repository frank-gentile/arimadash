#%%
import pandas as pd
from fredapi import Fred
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA


fred = Fred(api_key = '13a856ab4d9b2976146eb2e1fddd511d')

df = pd.Series(fred.get_series('CPIAUCSL').dropna(),name='CPI')

df2 = pd.Series(fred.get_series('CBETHUSD',frequency='w').dropna(),name='ETH')

combined = pd.merge(df,df2,how='outer',left_index=True,right_index=True)
combined.index.name='Date'
# combined.to_csv('data.csv')
# %%
import pandas as pd
from fredapi import Fred
from pandas import datetime
option_slctd = 'CPI'
data = pd.read_csv('data.csv')
data = data.set_index('Date')
df = data[option_slctd]
df = pd.DataFrame(df).dropna()
df.index = pd.DatetimeIndex(df.index.values,freq='MS')
start_date = datetime(2003,1,1)
end_date = datetime(2021,1,1)
df=df[df.index>start_date]
df=df[df.index<end_date]
p_slctd = 2
q_slctd = 2
diff_slctd = 1

def CreatePredictions(p,q,d,test,train):
    predictions = list()
    for t in range(len(test)+1):
        model = ARIMA(train, order=(p,q,d))
        model_fit = model.fit()
        output = model_fit.forecast()
        residuals = pd.DataFrame(model_fit.resid)
        yhat = output.iloc[0]
        predictions.append(yhat)
        try:
            obs = test.iloc[t]
            train = train.append(obs)
        except:
            pass
    pred = pd.DataFrame(predictions,index=test.index.union([test.index.shift(1)[-1]]))
    residuals = residuals.iloc[q:]
    return pred, residuals

size = int(len(df) * 0.75)
train, test = df[0:size], df[size:]
pred, residuals = CreatePredictions(int(p_slctd),int(q_slctd),int(diff_slctd),test,train)
# %%
