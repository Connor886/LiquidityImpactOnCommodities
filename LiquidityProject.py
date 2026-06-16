import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as pp
from pathlib import Path
csv_path =  Path(__file__).parent / 'CLc1_20260105.csv'
data= pd.read_csv(csv_path)
data= data.dropna(axis=1, how='all')
data['Date-Time']= pd.to_datetime(data['Date-Time'])
quotes= data[(data['Type']=='Quote')&(data['Ask Price']>data['Bid Price'])].copy().dropna(axis=1, how='all').sort_values('Date-Time')
trades= data[data['Type']=='Trade'].copy().dropna(subset=['Price']).dropna(axis=1, how='all').sort_values('Date-Time')
agg= '10min'
liq= dict()
all_valuations=[
                'mid','abs_spread','relative_spread','depth','dollar_depth','order_imbalance','log_depth','quote_slope',
                'volume','dollar_volume',
                'effective_spread','relative_effective_spread','relative_effective_spread_last',
                'amivest',
                'trade_count','relative_effective_spread_agg','relative_effective_spread_last_agg','volume_agg','dollar_volume_agg','mid_agg','mid_return','abs_mid_return','flow_ratio','amihud','depth_agg','dollar_depth_agg','relative_spread_agg','composite_liquidity','quote_slope_agg','order_imbalance_agg','order_ratio','excess_depth','absolute_dispersion_ratio','future_return','liquidation_time'
                ]

#filtered and merged trade and quote valuations
tq_merged= pd.merge_asof(trades[['Date-Time', 'Price', 'Volume']],
                         quotes[['Date-Time', 'Bid Price', 'Ask Price','Bid Size','Ask Size']],
                         on='Date-Time',
                         direction='backward',)
tq_merged.set_index('Date-Time',inplace=True)
tq_merged.index= tq_merged.index.tz_convert('America/Chicago')
tq_merged['mid']= (tq_merged['Bid Price']+tq_merged['Ask Price'])/2
liq['effective_spread']= 2*abs(tq_merged['Price']-tq_merged['mid'])
liq['relative_effective_spread']= liq['effective_spread']/tq_merged['mid']
liq['relative_effective_spread_last']= liq['effective_spread']/tq_merged['Price']

#set index to time series and switch time zones
quotes, trades= quotes.set_index('Date-Time'), trades.set_index('Date-Time')
quotes.index, trades.index= quotes.index.tz_convert('America/Chicago'),trades.index.tz_convert('America/Chicago')

#quote valuations
liq['mid']=(quotes['Bid Price']+quotes['Ask Price'])/2
liq['abs_spread']= quotes['Ask Price']-quotes['Bid Price']
liq['relative_spread']= liq['abs_spread']/liq['mid']
liq['depth']= quotes['Bid Size']+quotes['Ask Size']
liq['dollar_depth']= ((quotes['Bid Size']*quotes['Bid Price'])+(quotes['Ask Size']*quotes['Ask Price']))/2
liq['order_imbalance']= (quotes['Bid Size']-quotes['Ask Size'])/(quotes['Bid Size']+quotes['Ask Size'])
liq['log_depth']= np.log(quotes['Bid Size']*quotes['Ask Size'])
liq['quote_slope']= liq['abs_spread']/liq['log_depth']

#trade valuations
liq['volume']= trades['Volume'].copy()
liq['dollar_volume']= trades['Price']*liq['volume']

#help calculate aggregates valuations with 1 min intervals
ask_size= quotes['Ask Size'].resample(agg).last()
ask_price= quotes['Ask Price'].resample(agg).last()
bid_size= quotes['Bid Size'].resample(agg).last()
bid_price= quotes['Bid Price'].resample(agg).last()
liq['abs_spread_agg'] = liq['abs_spread'].resample(agg).mean()
liq['log_depth_agg'] = liq['log_depth'].resample(agg).mean()

#aggregate valuations with 1 min intervals
liq['trade_count']= trades['Price'].resample(agg).count()
liq['relative_effective_spread_agg']= liq['relative_effective_spread'].resample(agg).mean()
liq['relative_effective_spread_last_agg'] = liq['relative_effective_spread_last'].resample(agg).mean()
liq['volume_agg']= trades['Volume'].resample(agg).sum() 
liq['dollar_volume_agg']= (trades['Price']*trades['Volume']).resample(agg).sum()
liq['mid_agg']= liq['mid'].resample(agg).last()
liq['mid_return']= np.log(liq['mid_agg']).diff()
liq['abs_mid_return']= abs(liq['mid_return'])
liq['flow_ratio']= liq['trade_count']*liq['dollar_volume_agg']
liq['amihud']= liq['abs_mid_return']/liq['dollar_volume_agg']
liq['depth_agg']= liq['depth'].resample(agg).mean()
liq['dollar_depth_agg']= liq['dollar_depth'].resample(agg).mean()
liq['relative_spread_agg']= liq['relative_spread'].resample(agg).last()
liq['composite_liquidity']= liq['relative_spread_agg']/liq['dollar_depth_agg']
liq['quote_slope_agg'] = liq['abs_spread_agg']/liq['log_depth_agg']
liq['order_imbalance_agg']= liq['order_imbalance'].resample(agg).mean()
liq['order_ratio']= abs(ask_size-bid_size)/liq['dollar_volume_agg']
liq['excess_depth']= abs(ask_size-bid_size)
liq['absolute_dispersion_ratio']= abs(1-(((bid_size*bid_price)+(ask_size*ask_price))/((bid_size+ask_size)*liq['mid_agg'])))
liq['future_return']= np.log(trades['Price'].resample(agg).last()).diff().shift(-1)

#causing mathematical issues when no return
liq['amivest']= liq['dollar_volume_agg']/liq['abs_mid_return']

#Liquidation time metric using aggressive sell
raw_volume= data.set_index('Date-Time')[['Volume']]
raw_bid= data.set_index('Date-Time')[['Bid Size']]
bid_agg= raw_bid.resample(agg).last()
y= raw_bid.median().iloc[0]
def liquidation_time(volume,bid,y):
    diff= []
    for i in bid.index:
        start= i
        resid= y-bid.loc[start,'Bid Size']
        cumsum= volume.loc[start:].cumsum()
        end= cumsum[cumsum['Volume']>=resid].index[0]
        diff.append(np.log((end-start).total_seconds()))
    return pd.DataFrame({'time':diff})
liq['liquidation_time']= liquidation_time(raw_volume,bid_agg,y)['time']
liq['liquidation_time'].index= liq['trade_count'].index

#separate aggregate metrics into one dataframe and trim nans
aggregates= pd.DataFrame({})
for i in all_valuations[14:]:
    aggregates[i]= liq[i]
aggregates= aggregates.iloc[1:-1,:]

#linear regression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#features based off the papers
x = aggregates[['relative_effective_spread_last_agg','flow_ratio','dollar_depth_agg','volume_agg','amihud','order_ratio','liquidation_time']]
y= aggregates[['future_return']]
g_mse= []
g_ar2= []
model= LinearRegression()
for i in range(1000):
    xt, xs, yt, ys= train_test_split(x, y, test_size=0.2)
    model.fit(xt,yt)
    predict= model.predict(xs)
    mse= mean_squared_error(ys, predict)
    n= xs.shape[0]
    k= x.shape[1]
    r2= model.score(xs,ys)
    ar2= 1-(((1-r2)*(n-1))/(n-k-1))
    g_mse.append(mse)
    g_ar2.append(ar2)
