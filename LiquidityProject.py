import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as pp, arch
data= pd.read_csv('/Users/gregoriogarcia/Downloads/CLc1_20260105.csv')
data= data.dropna(axis=1, how='all')
data['Date-Time']= pd.to_datetime(data['Date-Time'])
quotes= data[(data['Type']=='Quote')&(data['Ask Price']>data['Bid Price'])].copy().dropna(axis=1, how='all').sort_values('Date-Time')
trades= data[data['Type']=='Trade'].copy().dropna(subset=['Price']).dropna(axis=1, how='all').sort_values('Date-Time')
agg= '10min'
liq= dict()
all_valuations=[
                'mid','mid_return','abs_mid_return','abs_spread','log_abs_spread','relative_spread','relative_spread_log','log_relative_spread','depth','dollar_depth','order_imbalance','log_depth','quote_slope',
                'direction','volume','dollar_volume','signed_volume','signed_dollar_volume',
                'relative_spread_last','effective_spread','relative_effective_spread','relative_effective_spread_last','log_effective_spread','percent_effective_spread','signed_effective_spread','signed_percent_effective_spread',
                'log_quote_slope','adjusted_log_quote_slope','composite_liquidity',
                'trade_count','volume_1min','dollar_volume_1min','mid_1min','mid_return_1min','abs_mid_return_1min','flow_ratio','amihud','amivest','depth_1min','dollar_depth_1min','relative_spread_1min','composite_liquidity_1min','quote_slope_1min','order_imbalance_1min','order_ratio','excess_depth','absolute_dispersion_ratio','liquidation_time'
                ]
#filtered and merged trade and quote valuations 
trades['direction']= trades['Aggressive Order Condition'].str.strip().map({'ASK':1,'BID':-1})
tq_merged= pd.merge_asof(trades[['Date-Time', 'Price', 'Volume', 'direction']],
                         quotes[['Date-Time', 'Bid Price', 'Ask Price','Bid Size','Ask Size']],
                         on='Date-Time',
                         direction='backward',)
tq_merged.set_index('Date-Time',inplace=True)
tq_merged.index= tq_merged.index.tz_convert('America/Chicago')
tq_merged['mid']= (tq_merged['Bid Price']+tq_merged['Ask Price'])/2
tq_signed= tq_merged.dropna(subset=['direction'])
liq['relative_spread_last']= (tq_merged['Ask Price']-tq_merged['Bid Price'])/tq_merged['Price']
liq['effective_spread']= 2*abs(tq_merged['Price']-tq_merged['mid'])
liq['relative_effective_spread']= liq['effective_spread']/tq_merged['mid']
liq['relative_effective_spread_last']= liq['effective_spread']/tq_merged['Price']
liq['signed_effective_spread']= 2*(tq_signed['Price']-tq_signed['mid'])*tq_signed['direction']
liq['signed_percent_effective_spread']= 2*((tq_signed['Price']-tq_signed['mid'])/tq_signed['mid'])*tq_signed['direction']
#set index to time series and switch time zones
quotes, trades= quotes.set_index('Date-Time'), trades.set_index('Date-Time')
quotes.index, trades.index= quotes.index.tz_convert('America/Chicago'),trades.index.tz_convert('America/Chicago')
#quote valuations
liq['mid']=(quotes['Bid Price']+quotes['Ask Price'])/2
liq['mid_return']= np.log(liq['mid']).diff()
liq['abs_mid_return']= abs(liq['mid_return'])
liq['abs_spread']= quotes['Ask Price']-quotes['Bid Price']
liq['relative_spread']= liq['abs_spread']/liq['mid']
liq['relative_spread_log']= np.log(quotes['Ask Price'])-np.log(quotes['Bid Price'])
liq['depth']= quotes['Bid Size']+quotes['Ask Size']
liq['dollar_depth']= ((quotes['Bid Size']*quotes['Bid Price'])+(quotes['Ask Size']*quotes['Ask Price']))/2
liq['order_imbalance']= (quotes['Bid Size']-quotes['Ask Size'])/(quotes['Bid Size']+quotes['Ask Size'])
liq['log_depth']= np.log(liq['depth'])
liq['quote_slope']= liq['abs_spread']/liq['log_depth']
#trade valuations
liq['direction']= trades['Aggressive Order Condition'].str.strip().map({'ASK':1,'BID':-1})
liq['volume']= trades['Volume'].copy()
liq['dollar_volume']= trades['Price']*liq['volume']
liq['signed_volume']= (liq['direction']*liq['volume']).dropna()
liq['signed_dollar_volume']= (liq['direction']*liq['dollar_volume']).dropna()
#valuations made with a combo of all others made
liq['log_quote_slope']= liq['relative_spread_log']/liq['log_depth']
liq['adjusted_log_quote_slope']= liq['log_quote_slope']*(1+abs(np.log(quotes['Ask Size']/quotes['Bid Size'])))
liq['composite_liquidity']= liq['relative_spread']/liq['dollar_depth']
#aggregate valuations with 1 min intervals
liq['trade_count']= trades['Price'].resample(agg).count()
liq['volume_agg']= trades['Volume'].resample(agg).sum() 
liq['dollar_volume_agg']= (trades['Price']*trades['Volume']).resample(agg).sum()
liq['mid_agg']= liq['mid'].resample(agg).last()
liq['mid_return_agg']= np.log(liq['mid_1min']).diff()
liq['abs_mid_return_agg']= abs(liq['mid_return_1min'])
liq['flow_ratio']= liq['trade_count']/liq['dollar_volume_1min']
liq['amihud']= liq['abs_mid_return_1min']/liq['dollar_volume_1min']
liq['amivest']= liq['dollar_volume_1min']/liq['abs_mid_return_1min']
liq['depth_agg']= liq['depth'].resample(agg).mean()
liq['dollar_depth_agg']= liq['dollar_depth'].resample(agg).mean()
liq['relative_spread_agg']= liq['relative_spread'].resample(agg).mean()
liq['composite_liquidity_agg']= liq['relative_spread_1min']/liq['dollar_depth_1min']
liq['quote_slope_agg']= liq['quote_slope'].resample(agg).mean()
liq['order_imbalance_agg']= liq['order_imbalance'].resample(agg).mean()
liq['order_ratio']= abs(quotes['Ask Size'].resample(agg).mean()-quotes['Bid Size'].resample(agg).mean())/liq['dollar_volume_1min']
liq['excess_depth']= abs(quotes['Ask Size'].resample(agg).last()-quotes['Bid Size'].resample(agg).last())
liq['absolute_dispersion_ratio']= abs(1-(((quotes['Bid Size'].resample(agg).last()*quotes['Bid Price'].resample(agg).last())+(quotes['Ask Size'].resample(agg).last()*quotes['Ask Price'].resample(agg).last()))/((quotes['Bid Size'].resample(agg).last()+quotes['Ask Size'].resample(agg).last())*liq['mid_1min'])))
#tbd
#liq['liquidity_intensity']=
#liq['hyper_plane']=
#Linear Regression using liquidation time
raw_volume= data.set_index('Date-Time')[['Volume']].dropna()
raw_bid= data.set_index('Date-Time')[['Bid Size']].dropna()
bid_1min= raw_bid.resample(agg).last()
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
liq['liquidation_time']= liquidation_time(raw_volume,bid_1min,y)['time']
liq['liquidation_time'].index= liq['trade_count'].index
aggregates= pd.DataFrame({})
for i in all_valuations[29:]:
    aggregates[i]= liq[i]
price= data[['Price','Date-Time']]
price.set_index('Date-Time',inplace=True)
price= price.tz_convert('America/Chicago')
price= price.resample(agg).last()
aggregates['price']= price['Price']
aggregates['future_return_1min']= aggregates.price.pct_change().shift(-1)
#print(agg)
#print(aggregates.corr().future_return_1min)
#print('length',len(aggregates))
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x= aggregates[['absolute_dispersion_ratio','order_ratio','quote_slope_1min','amivest','amihud','volume_1min','flow_ratio','depth_1min','composite_liquidity_1min','liquidation_time']]
y= aggregates[['future_return_1min']]
g_mse= []
model= LinearRegression()
for i in range(1000):
    xt, xs, yt, ys= train_test_split(x.iloc[:-1,:], y.iloc[:-1], test_size=0.2)
    model.fit(xt,yt)
    predict= model.predict(xs)
    mse= mean_squared_error(ys, predict)
    g_mse.append(mse)
'''no breaks
    quantiles for distribution for aggregation across months'''







