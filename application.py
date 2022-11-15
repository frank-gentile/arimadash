# -*- coding: utf-8 -*-
import pandas as pd # Library that manages dataframes
from statsmodels.tsa.arima.model import ARIMA
from pandas import datetime
from datetime import date
import plotly.graph_objs as go
from statsmodels.graphics.tsaplots import pacf, acf
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import dash  # (version 1.12.0) pip install dash
# import dash_core_components as dcc
# import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_loading_spinners as dls
from dash.dependencies import Input, Output, State
import math
import plotly.figure_factory as ff


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
application = app.server
app.title = 'Creating an ARIMA model in Dash'
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

# %%
 #Dashboard starts here
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([
    html.Br(),
    dbc.Row(dbc.Col(html.H1("How to create an ARIMA model "),width={'size':'auto'}),align='center',justify='center'),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Markdown(
        '''Hello and welcome! The purpose of this dashboard is instructional, to show how to create an ARIMA model to forecast inflation. This might be
        used as a starting place for those first learning about [ARIMA models](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average), or how to get started. ARIMA 
        stands for Autoregressive (AR), Integrated (I), Moving Average (MA) model. But what does that mean?
        Autoregressive models generally use past behavior to predict future behavior, Moving Average models use past random error terms to predict future 
        error terms, and Integrated refers to a series that is differenced. To difference is to take an observation and subtract the previous observation from it, or y(t) - y(t-1).
        This is done to detrend, or transform a nonstationary series into a stationary one. In general, stationary models are easier to predict since by definition, observations revert back to the mean. Surprisingly, 
        a simple ARIMA model doesn't do too bad of a job predicting inflation, and has been used before in various studies [(Meyler et al 1998)](https://www.researchgate.net/publication/23543270_Forecasting_irish_inflation_using_ARIMA_models), 
        [(Kuhe et al 2016)](https://www.researchgate.net/publication/337673205_Modeling_and_Forecasting_CPI_Inflation_in_Nigeria_Application_of_Autoregressive_Integrated_Moving_Average_Homoskedastic_Model), 
        [(Junttila 2001)](https://econpapers.repec.org/article/eeeintfor/v_3a17_3ay_3a2001_3ai_3a2_3ap_3a203-230.htm). The approach taken here follows [Box Jenkins method](https://en.wikipedia.org/wiki/Box–Jenkins_method), 
        and as such it's necessary that the series is stationary before continuing. Feel free to interact and crete your model!''',link_target="_blank"
    ),width=11,align='center'),align='center',justify='center'),
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Dropdown(id="slct_dataset",
                        options=[
                            {"label": "CPI", "value": "CPI"},
                            {'label':'ETH','value':'ETH'}],
                        multi=False,
                        value="CPI",
                        style={'width': "60%"}
                        )]),align='center',width={'size':3}),
        dbc.Col([
            html.Label('Number of times differenced d='),
            dcc.Input(id='slct_diff',value=1)],align='center',width={'size':4}),
        dbc.Col([
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=datetime(1947,1,1),
                max_date_allowed=datetime.today(),
                start_date=datetime(2003,1,1),
                end_date=datetime(2022,11,1)
                )],align='center',width={'size':3})
    ],align='center',justify='center'),
    html.Br(),
    dbc.Row(dbc.Col(dbc.Button('Update', id='submit-val', n_clicks=0,color='primary'),width=11,align='center'),align='center',justify='center'),

                
    dbc.Row(dbc.Col(dls.Hash(dcc.Graph(id='Data_Graph',figure={}),
                        color="#435278",
                        speed_multiplier=2,
                        size=100,
                    ))),
    dbc.Row([
        dbc.Col(html.H5([
            html.Div(id='ADF',children=[]),
            ]),width=6,align='center'),
        dbc.Col(html.H5([
            html.Div(id='KPSS',children=[]),
            ]),width=5,align='center')
            ],align='center',justify='center'),
    html.Br(),
    
    dbc.Row([dbc.Col(dcc.Markdown('''Here the [Augmented Dickey Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey–Fuller_test) 
        tells us if the series is stationary or not. Nonstationary 
        series are much more difficult to forecast, since observations can diverge from the mean. 
        In the test, a more negative critical value gives a higher 
        confidence level in the rejection of the null of non-stationarity.
        So if the test stat is more negative than the critical value we conclude the series is stationary.
        However, we don't want the model to lose predictive power by differencing too many times [(Cochrane 2018)](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5ee12618da7ad1571d6a9ce7/1591813656542/overdifferencing.pdf)
        so you should only difference as many times as is necessary to pass the tests.''',link_target="_blank"),width=6,align='center'),
        dbc.Col(dcc.Markdown('''There exists a problem with the ADF test in that if a series is borderline between stationary
        and nonstationary, it may give the wrong answer since there is inconclusive evidence. To double check, we can perform a 
        [KPSS test](https://en.wikipedia.org/wiki/KPSS_test) for stationarity. Here, if the test statistic is less than the critical value, there is not enough evidence to reject the null 
        of stationarity, so we cannot conclude that it is not stationary. Here, we expect the series to be stationary and so it should have a large p value, but for this 
        particular test the the p value is bounded from 0.01 to 0.1, so p values might be large, but will return as 0.1 and smaller will return as 0.01. 
        ''',link_target="_blank"),width={'size':5})],align='center',justify='center'),
    
    html.Br(),


    dbc.Row([dbc.Col(dcc.Graph(id='ACF',figure={},style={'display':'inline-block'}),width=6,align='center'),
             dbc.Col(dcc.Graph(id='PACF',figure={},style={'display':'inline-block'}),width=6,align='left')],align='center',justify='center'),
    
    dbc.Row(dbc.Col(dcc.Markdown(
        ''' The next step is to consider the Autocorrelation and Partial Autocorrelation plots. Here we choose the number of moving average
        terms (q) based on the PACF plot. The order is equal to the number of lags, typically only the first few, that are outside the 
        green region (95% confidence interval). Similarly, we consider the ACF for the order (p) of the AR component, or how many 
        AR terms we should include. '''
    ),width=11,align='center'),align='center',justify='center'),
    
    dbc.Row(dbc.Col(html.Div([html.Label('Order p='),
        dcc.Input(id='slct_p',value=1)]),width=11),align='center',justify='center'),

    dbc.Row(dbc.Col(html.Div([html.Label('Order q='),
        dcc.Input(id='slct_q',value=2)]),width=11),align='center',justify='center'),
    
    html.Br(),
    dbc.Row(dbc.Col([dbc.Button('Create Model',id='button2',n_clicks=0,color='primary'),
    html.Br(),
    html.Div(id='output_container2', children=[])],width=11),align='center',justify='center'),
    html.Br(),
    dbc.Row(dbc.Col([dls.Hash(dcc.Graph(id='Prediction_Graph', figure={}),
                        color="#435278",
                        speed_multiplier=2,
                        size=100,
                    )],width={'size':12}),align='center'),
    dbc.Row([dbc.Col(dcc.Graph(id='Residuals_Graph', figure={}),width=6),
            dbc.Col(dcc.Markdown('''Lastly, we want to make sure that errors are distributed normally. 
                     Typically, we would perform a [Jarque Bera](https://en.wikipedia.org/wiki/Jarque–Bera_test)
                      test for normality in residuals. However, results
                     may be inconsistent with fewer than 2000 datapoints, and since the inflation data is given monthly our models might have 
                     fewer datapoints than that. Looking at the distribution of residuals, we can see that for the default model the errors look 
                     pretty good, but skewed slightly left. This might be ok if you are comfortable with your model overpredicting more than 
                     it underpredicts, but in general we want our models to have normally distributed errors. To improve the model, we could 
                     consider a log transformation, but this might vary depending on the data.''',link_target="_blank"
                     ),width={'size':5})],align='center'),
    html.Br(),
    dbc.Row(dbc.Col(dcc.Markdown('''Thanks for reading! If you want to contact me with questions or comments 
        you can do so at [franky.gentile@gmail.com](mailto:franky.gentile@gmail.com)''',link_target="_blank"),width=11,align='center'),align='center',justify='center'),

    html.Br(),
    html.Br()

])





# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id="ACF", component_property="figure"),
     Output(component_id="PACF", component_property="figure"),
     #Output(component_id='d', component_property='children'),
     Output(component_id='ADF', component_property='children'),
     Output(component_id='Data_Graph',component_property='figure'),
     Output(component_id='KPSS',component_property='children')],
     [Input('submit-val','n_clicks')],
     [State('slct_dataset','value'),
            State('slct_diff','value'),
            State('my-date-picker-range','start_date'),
            State('my-date-picker-range','end_date')]
)
def update_acf_graph(n_clicks,option_slctd,diff_slctd,start_date,end_date):
    #d = "d={}".format(diff_slctd)
#   date_range =  state_date={} end_date={}.format(start_date,end_date)
#update this to take the difference in months rather than calling API
    data = pd.read_csv('data.csv')
    data = data.set_index('Date')
    df = data[option_slctd]
    # if option_slctd == 'CPI':
    #     tag = 'CPIAUCSL'
    #     df = pd.Series(fred.get_series(tag).dropna(),name='CPI')
    # elif option_slctd == 'ETH':
    #     tag = 'CBETHUSD'
    #     df = pd.Series(fred.get_series(tag,frequency='w').dropna(),name='ETH')
    
    df = pd.DataFrame(df).dropna()
    df.index = pd.DatetimeIndex(df.index.values)
    df=df[df.index>start_date]
    df=df[df.index<end_date]
    size = int(len(df) * 0.75)
    train, test = df[0:size], df[size:]

    for i in range(int(diff_slctd)):
        df = df.diff().dropna()
    df_pacf = pacf(df)
    df_acf = acf(df)
    adf_test = adfuller(df)
    if int(diff_slctd)>0:
        kpss_test = kpss(df)
    else:
        kpss_test = kpss(df,regression='ct')
    kpss_str = 'KPSS Test Statistic: {} with a p-value of {} and critical values of {} (1%) {} (5%) and {} (10%)'.format(
        kpss_test[0].round(2), kpss_test[1],round(kpss_test[3]['1%'],2),round(kpss_test[3]['5%'],2),round(kpss_test[3]['10%'],2))
    adf_str = 'ADF Test Statistic: {} with a p-value of {} and critical values of {} (1%) {} (5%) and {} (10%)'.format(
        adf_test[0].round(2), adf_test[1].round(3),adf_test[4]['1%'].round(2),adf_test[4]['5%'].round(2),adf_test[4]['10%'].round(2))
    stationary_plot = go.Figure(go.Scatter(x=df.index,y=df[option_slctd]))
    stationary_plot.update_layout(title='Series with d={}'.format(diff_slctd),
        xaxis_title='Date')

    lower95 = -1.96/math.sqrt(len(df))
    upper95 = -lower95

    fig2= go.Figure()
    fig2.add_trace(go.Bar(
        x= np.arange(len(df_acf)),
        y= df_acf,
        name= 'ACF',
        width=[0.2]*len(df_acf),
        showlegend=False
        ))
    fig2.add_hrect(y0=lower95, y1=upper95,line_width=0,fillcolor='green',opacity=0.1)
    fig2.add_trace(go.Scatter(
        mode='markers',
        x=np.arange(len(df_acf)),
        y= df_acf,
        marker=dict(color='blue',size=8),
        showlegend=False
    ))
    fig2.update_layout(
        title="Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        #     autosize=False,
        #     width=500,
            height=500,
        )

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x= np.arange(len(df_pacf)),
        y= df_pacf,
        name= 'PACF',
        width=[0.2]*len(df_pacf),
        showlegend=False
        ))
    fig3.add_hrect(y0=lower95, y1=upper95,line_width=0,fillcolor='green',opacity=0.1)
    fig3.add_trace(go.Scatter(
        mode='markers',
        x=np.arange(len(df_pacf)),
        y= df_pacf,
        marker=dict(color='blue',size=8),
        showlegend=False
    ))
    fig3.update_layout(
        title="Partial Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        #     autosize=False,
        #     width=500,
            height=500,
        )

    return fig2, fig3, adf_str,stationary_plot, kpss_str
@app.callback(
    [Output(component_id='Prediction_Graph', component_property='figure'),
    Output(component_id='output_container2', component_property='children'),
    Output(component_id='Residuals_Graph',component_property='figure')
    ],
    [Input('button2','n_clicks')],
     [State('slct_dataset','value'),
            State('slct_diff','value'),
            State('slct_p','value'),
            State('slct_q','value'),
            State('my-date-picker-range','start_date'),
            State('my-date-picker-range','end_date')])

def create_model_graph(n_clicks,option_slctd,diff_slctd,p_slctd,q_slctd,start_date,end_date):
    container = "d={} p={} q={} state_date={} end_date={}".format(diff_slctd,p_slctd,q_slctd,start_date,end_date)

    data = pd.read_csv('data.csv')
    data = data.set_index('Date')
    df = data[option_slctd]
    # if option_slctd == 'CPI':
    #     tag = 'CPIAUCSL'
    #     df = pd.Series(fred.get_series(tag).dropna(),name='CPI')
    # elif option_slctd == 'ETH':
    #     tag = 'CBETHUSD'
    #     df = pd.Series(fred.get_series(tag,frequency='w').dropna(),name='ETH')
    
    df = pd.DataFrame(df).dropna()
    if option_slctd == 'CPI':
        frq = 'MS'
    elif option_slctd == 'ETH':
        frq = 'W-FRI'
    df.index = pd.DatetimeIndex(df.index.values,freq=frq)
    df=df[df.index>start_date]
    df=df[df.index<end_date]
    size = int(len(df) * 0.75)
    train, test = df[0:size], df[size:]
    pred, residuals = CreatePredictions(int(p_slctd),int(q_slctd),int(diff_slctd),test,train)
    err = (pred[0]-test[option_slctd])**2
    mse = np.mean(err[0]).round(2)

    D1 = go.Scatter(x=train.index,y=train[option_slctd],name = 'Train Actual') # Training actuals
    D2 = go.Scatter(x=test.index,y=test[option_slctd],name = 'Test Actual') # Testing actuals
    D3 = go.Scatter(x=pred.index,y=pred[0],name = 'Prediction') # Testing predction

    # Combine in an object  
    line = {'data': [D1,D2,D3],
            'layout': {
                'xaxis' :{'title': 'Date'},
                'yaxis' :{'title': option_slctd},
                'title' : option_slctd
            }}
    fig = go.Figure(line)
    fig.update_layout(height=500)

    fig2 = ff.create_distplot([residuals[0].values], ['Residuals'],show_hist=False)
    fig2.update_layout(xaxis_title='Residual',yaxis_title='KDE Probability Density',title='Mean squared error = '+str(mse))

    return fig, container,fig2
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
