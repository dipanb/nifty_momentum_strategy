#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import requests


# In[2]:


nifty500_scrips = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/ind_nifty500list.csv')
nifty500_scrips = nifty500_scrips['Symbol'] + ".NS"


# In[3]:


start = datetime.datetime(2010,1,1)
end = datetime.datetime(2024,3,31)
price_data = yf.download(nifty500_scrips.to_list(), start=start, end=end)['Adj Close']


# In[4]:


columns_to_drop = price_data.columns[price_data.isna().sum()/price_data.shape[0]>.1].to_list()
price_data = price_data.drop(columns=columns_to_drop)


# In[5]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

def daily_weekly_rollup(daily_returns):
    weekly_returns = (1 + daily_returns).resample('W').prod() - 1
    return weekly_returns

def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

def weekly_monthly_rollup(weekly_returns):
    monthly_returns = (1 + weekly_returns).resample('M').prod() - 1
    return monthly_returns


# In[6]:


daily_returns = convert_prices_to_returns(price_data)
daily_returns[daily_returns > 0.2] = 0
weekly_returns = daily_weekly_rollup(daily_returns)
monthly_returns = daily_monthly_rollup(daily_returns)


# In[7]:


def return_lookback(returns,lookback_period):
    lookback_returns = (1 + returns).rolling(window=lookback_period).apply(lambda x: x.prod()) - 1
    return lookback_returns

def timeframe_high_low_ratio(prices, lookback, timeframe):
    if timeframe=='weekly':
        lookback = lookback * 5
    if timeframe=='monthly':
        lookback = lookback * 20
    high52_ratio = prices/prices.rolling(window=lookback).max()
    low52_ratio = prices/prices.rolling(window=lookback).min()
    return high52_ratio, low52_ratio

def SMAf_SMAs(prices,fast,slow):
    SMAf = prices.rolling(window=fast).mean()
    SMAs = prices.rolling(window=slow).mean()
    SMAf_s = (SMAf-SMAs)/SMAs
    return SMAf_s

def Periods_Up_Down(returns,lookback_period):
    Up_Down = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(returns.shape[0]):
        if i<(lookback_period-1):
            Up_Down.iloc[i] = 0
            continue
        else:
            temp_df = returns.iloc[(i-lookback_period+1):(i+1)]    
            for cols in temp_df.columns:
                pos = 0
                neg = 0
                for j in range(temp_df.shape[0]):
                    if temp_df[cols].iloc[j]>0:
                        pos+=1
                    else:
                        neg+=1
                Up_Down.iloc[i][cols] = pos - neg  
    return Up_Down


# In[8]:


def month_end_values(df):
    month_end_df = df.resample('M').last()
    return month_end_df

def rank_fn(indicator,order=True):
    asc_flag = True if order=='Ascending' else False
    rank_df = indicator.rank(axis=1, ascending = asc_flag)
    return rank_df

def select_top_x_bottom_x(rank,x):
    top_x = rank.apply(lambda row: row.nsmallest(x).index.tolist(), axis=1)
    bottom_x = rank.apply(lambda row: row.nlargest(x).index.tolist(), axis=1)   
    return top_x, bottom_x

def lookback_returns_long(returns,top_x,x):
    cols = []
    for i in range(x):
        cols.append(f'Long{i+1}')
    lookback_return_l_s = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(returns.shape[0]-1):
        for j in range(x):
                lookback_return_l_s.iloc[i,j] = returns.iloc[i][top_x[i][j]]
    return lookback_return_l_s

def return_series_long(returns,top_x,x,next_per=1):
    cols = []
    for i in range(x):
        cols.append(f'Long{i+1}')
    return_l_s = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(return_l_s.shape[0]-next_per):
        for j in range(x):
                return_l_s.iloc[i,j] = (1+returns.iloc[(i+1):(i+next_per+1)][top_x[i][j]]).prod()-1
    return return_l_s

def equally_wtd(returns_l_s, holding_period,friction = 0):
    return_next_period = returns_l_s.mean(axis=1) - friction
    return_next_period = pd.DataFrame(return_next_period)
    return_next_period.columns = ['returns']
    return return_next_period


# In[9]:


def calc_max_drawdown(return_series):
    comp_ret = pd.Series((return_series+1).cumprod())
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

#Analytics
def strategy_analytics(returns_alloc, timeframe, rf = 0.05):
    if timeframe=='Weekly':
        factor = 52
        length_returns = returns_alloc.shape[0]/52
    if timeframe=='Monthly':
        factor = 12
        length_returns = returns_alloc.shape[0]/12
    mean_return = returns_alloc['returns'].mean()*factor
    cagr = ((1+returns_alloc['returns']).prod())**(1/length_returns)-1
    st_dev = returns_alloc['returns'].std()*(factor**0.5)
    sharpe_ratio = (mean_return - rf)/st_dev
    sortino_ratio = (mean_return - rf)/(returns_alloc['returns'].loc[returns_alloc['returns']<0].std()*(factor**0.5))
    max_drawd = - calc_max_drawdown(np.array(returns_alloc['returns']))
    calmar_ratio = mean_return/ max_drawd
    success_rate = sum(returns_alloc['returns']>0)/len(returns_alloc['returns'])
    average_up = returns_alloc['returns'].loc[returns_alloc['returns']>0].mean()
    average_down = returns_alloc['returns'].loc[returns_alloc['returns']<0].mean()
    return mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawd,calmar_ratio,success_rate,average_up,average_down


# In[10]:


#Inputs
Timeframe = 'Weekly'
Lookback = 38
number_stocks = 5
holding_period = 4


# In[11]:


if Timeframe == 'Weekly':
    resample_freq = 'W'
if Timeframe == 'Monthly':
    resample_freq = 'M'

lookback_returns = return_lookback(weekly_returns,Lookback)
periods_ud = Periods_Up_Down(weekly_returns,Lookback)
# weekly_prices = price_data.resample('W').last()
SMA_5_20 = SMAf_SMAs(price_data,5,20).resample(resample_freq).last()
SMA_20_50 = SMAf_SMAs(price_data,20,50).resample(resample_freq).last()
# SMA_50_200 = SMAf_SMAs(price_data,50,200).resample(resample_freq).last()

High_ratio,Low_ratio  = timeframe_high_low_ratio(price_data,Lookback,'weekly')
High_ratio = High_ratio.resample('W').last().bfill()
Low_ratio = Low_ratio.resample('W').last().bfill()         


# In[12]:


rank_df = rank_fn(lookback_returns,"Descending")
rank_df_2 = rank_fn(periods_ud,"Descending")
rank_df_3 = rank_fn(High_ratio,"Descending")

# top_x, _ = select_top_x_bottom_x(rank_df,number_stocks)
top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2,number_stocks)
# top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2+rank_df_3,number_stocks)


# In[13]:


lookback_returns_l_s = lookback_returns_long(weekly_returns, top_x, number_stocks)
returns_l_s = return_series_long(weekly_returns, top_x, number_stocks,1)


# In[14]:


def trade_setup(returns, top_x, holding_period):
    returns_series = pd.Series(index=returns.index)
    cols = []
    for i in range(len(top_x[0])):
        cols.append(f'Long{i+1}')
    strategy_returns_long = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(0, returns.shape[0], holding_period):
        top_x_i = top_x[i]
        for j in range(i,i+holding_period):
            if j==(returns.shape[0]-1):
                break
            returns_series[j] = returns.loc[returns.index[j+1],top_x_i].mean(axis=0)
            strategy_returns_long.iloc[j] = returns.loc[returns.index[j+1],top_x_i].values
    returns_series = pd.DataFrame(returns_series, columns=['returns'])
    return returns_series, strategy_returns_long


# In[15]:


# returns_alloc = equally_wtd(returns_l_s, holding_period, friction = 0)[:-holding_period]
returns_alloc, _ = trade_setup(weekly_returns, top_x, 1)
returns_alloc = returns_alloc[Lookback:-1]
returns_alloc


# In[16]:


mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc,Timeframe)
strategy_df = pd.DataFrame(columns = ['Lookback','Allocation','Holding Period','Number of stocks',
                                     'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
                                     'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])

strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Allocation' : 'Equally Wtd Long',
                                   'Holding Period': holding_period,
                                   'Number of stocks' : number_stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
strategy_df


# In[17]:


def strategies(Lookback,number_of_stocks,holding_period):
    
    Timeframe = "Weekly"
    
    lookback_returns = return_lookback(weekly_returns,Lookback)
    periods_ud = Periods_Up_Down(weekly_returns,Lookback)

    High_ratio,Low_ratio  = timeframe_high_low_ratio(price_data,Lookback,'weekly')
    High_ratio = High_ratio.resample('W').last().bfill()
    Low_ratio = Low_ratio.resample('W').last().bfill()
        
    rank_df = rank_fn(lookback_returns,"Descending")
    rank_df_2 = rank_fn(periods_ud,"Descending")
    rank_df_3 = rank_fn(High_ratio,"Descending")

    top_x_1, _ = select_top_x_bottom_x(rank_df,number_of_stocks)
    top_x_12, _ = select_top_x_bottom_x(rank_df+rank_df_2,number_of_stocks)
    top_x_13, _ = select_top_x_bottom_x(rank_df+rank_df_3,number_of_stocks)
    top_x_123, _ = select_top_x_bottom_x(rank_df+rank_df_2+rank_df_3,number_of_stocks)
        
    returns_alloc_1, _ = trade_setup(weekly_returns, top_x_1, holding_period)
    returns_alloc_1 = returns_alloc_1[(Lookback-1):-1]
    returns_alloc_12, _ = trade_setup(weekly_returns, top_x_12, holding_period)
    returns_alloc_12 = returns_alloc_12[(Lookback-1):-1]
    returns_alloc_13, _ = trade_setup(weekly_returns, top_x_13, holding_period)
    returns_alloc_13 = returns_alloc_13[(Lookback-1):-1]
    returns_alloc_123, _ = trade_setup(weekly_returns, top_x_123, holding_period)
    returns_alloc_123 = returns_alloc_123[(Lookback-1):-1]
    
    strategy_df = pd.DataFrame(columns = ['Lookback','Rank','Holding Period','Number of stocks',
                                     'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
                                     'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc_1,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : 'Returns',
                                   'Holding Period': holding_period,
                                   'Number of stocks' : number_of_stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc_12,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : 'Returns + Periods_ud',
                                   'Holding Period': holding_period,
                                   'Number of stocks' : number_of_stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc_13,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : 'Returns + High_Ratio',
                                   'Holding Period': holding_period,
                                   'Number of stocks' : number_of_stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc_123,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : 'Returns + Periods_ud + High_Ratio',
                                   'Holding Period': holding_period,
                                   'Number of stocks' : number_of_stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    return strategy_df


# In[18]:


strategies(38,2,1)


# In[19]:


##### Permutation of strategies #########################################

# Lookback = np.arange(2, 50, 4)
# Number_of_stocks = np.arange(2,21,4)
# Holding_period = np.arange(1,6,2)

# strategy_df = pd.DataFrame(columns = ['Lookback','Rank','Holding Period','Number of stocks',
#                                      'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
#                                      'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])

# for i in Lookback:
#     for j in Number_of_stocks:
#         for k in Holding_period:
#             print(i,j,k)
#             strategy_df = pd.concat([strategy_df, strategies(i,j,k)], ignore_index = True) 
            
# strategy_df


# In[20]:


# strategy_df.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Strategies Performance.xlsx')


# In[21]:


def run_strategy(Lookback,Selection,Holding_Period,Number_of_Stocks):
    Timeframe = "Weekly"
    
    lookback_returns = return_lookback(weekly_returns,Lookback)
    periods_ud = Periods_Up_Down(weekly_returns,Lookback)

    High_ratio,Low_ratio  = timeframe_high_low_ratio(price_data,Lookback,'weekly')
    High_ratio = High_ratio.resample('W').last().bfill()
    Low_ratio = Low_ratio.resample('W').last().bfill()
    
    rank_df = rank_fn(lookback_returns,"Descending")
    rank_df_2 = rank_fn(periods_ud,"Descending")
    rank_df_3 = rank_fn(High_ratio,"Descending")

    if Selection == 'R':
        top_x, _ = select_top_x_bottom_x(rank_df,Number_of_Stocks)
    if Selection == 'R+U':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2,Number_of_Stocks)
    if Selection == 'R+H':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_3,Number_of_Stocks)
    if Selection == 'R+U+H':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2+rank_df_3,Number_of_Stocks)
        
    returns_alloc, strategy_returns_all = trade_setup(weekly_returns, top_x, Holding_Period)
    returns_alloc = returns_alloc[(Lookback-1):-1]
    strategy_returns_all = strategy_returns_all.iloc[(Lookback-1):-1]
    
    strategy_df = pd.DataFrame(columns = ['Lookback','Rank','Holding Period','Number of stocks',
                                     'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
                                     'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : Selection,
                                   'Holding Period': Holding_Period,
                                   'Number of stocks' : Number_of_Stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    print(strategy_df)
    
    return returns_alloc, lookback_returns, High_ratio, top_x, strategy_returns_all


# In[22]:


strategy_returns,_,_,_,_ = run_strategy(38,'R+H',1,2)


# In[23]:


np.log((1+strategy_returns).cumprod()).plot(figsize=(10, 6),legend=False)
plt.title('NAV')
plt.show()


# In[24]:


strategy_returns.plot(figsize=(10, 6),legend=False)
plt.title('NAV')
plt.show()


# In[25]:


# strategy_returns.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Best Strategies Returns.xlsx')


# In[26]:


Nifty50_data = yf.download('^NSEI', start=start, end=end)['Adj Close']
# N50_weekly_returns = convert_prices_to_returns(Nifty50_data)
N50_returns = Nifty50_data.pct_change()
N50_returns.iloc[0] = 0
N50_returns = daily_weekly_rollup(N50_returns)


# In[27]:


# strategy_returns = pd.merge(strategy_returns, N50_returns, how='left', left_index=True, right_index=True)
# strategy_returns.columns = ['Strategy','Nifty50']


# In[28]:


def SMAf_SMAs(prices,fast,slow):
    SMAf = prices.rolling(window=fast).mean()
    SMAs = prices.rolling(window=slow).mean()
    SMAf_s = (SMAf-SMAs)/SMAs
    return SMAf_s

# Nifty_SMA_5_20 = SMAf_SMAs(Nifty50_data,5,20)
# Nifty_SMA_20_50 = SMAf_SMAs(Nifty50_data,20,50)
# Nifty_SMA_50_200 = SMAf_SMAs(Nifty50_data,50,200)


# In[29]:


# closest_dates = []
# for d in strategy_returns.index:
#     closest_date = Nifty_SMA_5_20.index[np.abs(Nifty_SMA_5_20.index - d + datetime.timedelta(days=2)).argmin()]
#     closest_dates.append(closest_date)
# Nifty_SMA_5_20 = Nifty_SMA_5_20[closest_dates]
# Nifty_SMA_5_20.index = strategy_returns.index
# Nifty_SMA_20_50 = Nifty_SMA_20_50[closest_dates]
# Nifty_SMA_20_50.index = strategy_returns.index
# Nifty_SMA_50_200 = Nifty_SMA_50_200[closest_dates]
# Nifty_SMA_50_200.index = strategy_returns.index
# strategy_returns['Nifty_SMA_5_20'] = Nifty_SMA_5_20 
# strategy_returns['Nifty_SMA_20_50'] = Nifty_SMA_20_50
# strategy_returns['Nifty_SMA_50_200'] = Nifty_SMA_50_200
# Nifty_rolling_8week = N50_returns
# Nifty_rolling_26week = N50_returns
# Nifty_rolling_52week = N50_returns


# In[30]:


# strategy_returns.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Best Strategies Returns.xlsx')


# In[31]:


# Nifty_SMA_5 = Nifty50_data.rolling(window=5).mean()
Nifty_SMA_20 = Nifty50_data.rolling(window=20).mean()
# Nifty_SMA_50 = Nifty50_data.rolling(window=50).mean()
# Nifty_SMA_100 = Nifty50_data.rolling(window=100).mean()

closest_dates = []
for d in strategy_returns.index:
    closest_date = Nifty_SMA_20.index[np.abs(Nifty_SMA_20.index - d + datetime.timedelta(days=2)).argmin()]
    closest_dates.append(closest_date)

# Nifty_SMA_5 = Nifty_SMA_5[closest_dates]
Nifty_SMA_20 = Nifty_SMA_20[closest_dates]
# Nifty_SMA_50 = Nifty_SMA_50[closest_dates]
# Nifty_SMA_100 = Nifty_SMA_100[closest_dates]


# In[32]:


import math
# Nifty_SMA_5_slope = np.degrees(np.arctan(Nifty_SMA_5.pct_change()))
Nifty_SMA_20_slope = np.degrees(np.arctan(Nifty_SMA_20.pct_change()))
# Nifty_SMA_50_slope = np.degrees(np.arctan(Nifty_SMA_50.pct_change()))
# Nifty_SMA_100_slope = np.degrees(np.arctan(Nifty_SMA_100.pct_change()))


# In[33]:


# strategy_returns['Nifty_SMA_5'] = np.array(Nifty_SMA_5)
strategy_returns['Nifty_SMA_20'] = np.array(Nifty_SMA_20)
# strategy_returns['Nifty_SMA_50'] = np.array(Nifty_SMA_50)
# strategy_returns['Nifty_SMA_100'] = np.array(Nifty_SMA_100)

# strategy_returns['Nifty_SMA_5_slope'] = np.array(Nifty_SMA_5_slope)
strategy_returns['Nifty_SMA_20_slope'] = np.array(Nifty_SMA_20_slope)
# strategy_returns['Nifty_SMA_50_slope'] = np.array(Nifty_SMA_50_slope)
# strategy_returns['Nifty_SMA_100_slope'] = np.array(Nifty_SMA_100_slope)


# In[34]:


# strategy_returns.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Best Strategies Returns 2.xlsx')
# strategy_returns


# In[35]:


#Strategy Modification 1
#Calculate sma 20 slope...dont take below -0.45
#Dont take stocks if lookback return negative
#Analaze individual stock winners and losers
strategy_returns, lookback_returns, High_ratio, top_x, strategy_returns_all = run_strategy(38,'R+H',1,2)


# In[36]:


lookback_returns = lookback_returns.loc[strategy_returns.index]
High_ratio = High_ratio.loc[strategy_returns.index]
top_x = top_x.loc[strategy_returns.index]


# In[37]:


strategy_returns = pd.merge(strategy_returns, N50_returns, how='left', left_index=True, right_index=True)
strategy_returns.columns = ['Strategy','Nifty50']
Nifty_SMA_20 = Nifty50_data.rolling(window=20).mean()
closest_dates = []
for d in strategy_returns.index:
    closest_date = Nifty_SMA_20.index[np.abs(Nifty_SMA_20.index - d + datetime.timedelta(days=2)).argmin()]
    closest_dates.append(closest_date)
Nifty_SMA_20 = Nifty_SMA_20[closest_dates]
Nifty_SMA_20_slope = np.degrees(np.arctan(Nifty_SMA_20.pct_change()))
strategy_returns['Nifty_SMA_20'] = np.array(Nifty_SMA_20)
strategy_returns['Nifty_SMA_20_slope'] = np.array(Nifty_SMA_20_slope)


# In[38]:


strategy_returns['Top A'] = 0
strategy_returns['Top B'] = 0
strategy_returns['Lookback A'] = 0
strategy_returns['Lookback B'] = 0
strategy_returns['High Ratio A'] = 0
strategy_returns['High Ratio B'] = 0
strategy_returns['Strategy A'] = 0
strategy_returns['Strategy B'] = 0

for i in range(strategy_returns.shape[0]):
    strategy_returns['Top A'].iloc[i] = top_x[i][0]
    strategy_returns['Top B'].iloc[i] = top_x[i][1]
    
    strategy_returns['Lookback A'].iloc[i] = lookback_returns.iloc[i][top_x[i]].values[0]
    strategy_returns['Lookback B'].iloc[i] = lookback_returns.iloc[i][top_x[i]].values[1]
    
    strategy_returns['High Ratio A'].iloc[i] = High_ratio.iloc[i][top_x[i]].values[0]
    strategy_returns['High Ratio B'].iloc[i] = High_ratio.iloc[i][top_x[i]].values[1]
    
    strategy_returns['Strategy A'].iloc[i] = strategy_returns_all['Long1'].iloc[i]
    strategy_returns['Strategy B'].iloc[i] = strategy_returns_all['Long2'].iloc[i]


# In[39]:


strategy_returns['Nifty_SMA_20_slope'].iloc[0] = 0
strategy_returns['Strategy 1'] = [strategy_returns['Strategy'].iloc[i] if strategy_returns['Nifty_SMA_20_slope'].iloc[i]>-0.43 else 0 for i in range(strategy_returns.shape[0])]


# In[43]:


# strategy_returns.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Best Strategies Returns 2.xlsx')
# strategy_returns`


# In[75]:


import requests
from bs4 import BeautifulSoup
import re
from random import randint
from time import sleep

nifty500_scrips = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/ind_nifty500list.csv')
nifty500_scrips['MarketCap'] = 0

scrip = 0
# while scrip <= (nifty500_scrips.shape[0]-1):
for i in range(scrip,nifty500_scrips.shape[0]):
    print(i, " / ", nifty500_scrips.shape[0])
    url = 'https://www.screener.in/company/' + nifty500_scrips['Symbol'].iloc[i] + '/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    nifty500_scrips['MarketCap'].iloc[i] = float(re.findall(r'\d+\.*\d*', 
                                text[text.find("Market Cap"):(text.find("Market Cap")+60)].replace(",", ""))[0])
    sleep(1)
    
# url = 'https://www.screener.in/company/' + nifty500_scrips['Symbol'].iloc[287] + '/'
# print(nifty500_scrips.iloc[287])
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')
# text = soup.get_text()
# nifty500_scrips['MarketCap'].iloc[i] = float(re.findall(r'\d+\.*\d*', 
#                                 text[text.find("Market Cap"):(text.find("Market Cap")+60)].replace(",", ""))[0])
    


# In[97]:


# nifty500_scrips.to_excel('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Market Cap Scrape 9 May 2024.xlsx')
Market_Cap = weekly_returns.copy()
Market_Cap.iloc[-1] = [nifty500_scrips['MarketCap'].loc[nifty500_scrips['YF_Code']==i] for i in Market_Cap.columns] 
for i in range(len(Market_Cap)-2, -1, -1):
    Market_Cap.iloc[i] = Market_Cap.iloc[i+1]/(1+weekly_returns.iloc[i+1]) 


# In[104]:


def percentile_90(row):
    return np.percentile(row, 90)

MCap_90 = Market_Cap.apply(percentile_90, axis=1)


# In[110]:


def trade_setup_4_1(returns, top_x, lookback_returns, High_ratio, Market_Cap, MCap_90, holding_period=1):
    returns_series = pd.Series(index=returns.index)
    cols = []
    for i in range(len(top_x[0])):
        cols.append(f'Long{i+1}')
    strategy_returns_long = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(0, returns.shape[0], holding_period):
        top_x_i = top_x[i]
        for j in range(i,i+holding_period):
            if j==(returns.shape[0]-1):
                break
            lookback_returns_A = lookback_returns.loc[returns.index[j],top_x_i[0]]
            lookback_returns_B = lookback_returns.loc[returns.index[j],top_x_i[1]]
            high_ratio_A = High_ratio.loc[returns.index[j],top_x_i[0]]
            high_ratio_B = High_ratio.loc[returns.index[j],top_x_i[1]]
            
            MCap_90_i = MCap_90[j]
            Market_Cap_A = Market_Cap.loc[returns.index[j],top_x_i[0]]
            Market_Cap_B = Market_Cap.loc[returns.index[j],top_x_i[1]]
            
            lookback_flag_A = 1 if lookback_returns_A>0.9 else 0
            lookback_flag_B = 1 if lookback_returns_B>0.71 else 0
            high_flag_A = 1 if high_ratio_A>0.976 else 0
            high_flag_B = 1 if high_ratio_B>0.962 else 0
            MCap_flag_A = 1 if Market_Cap_A<MCap_90_i else 0
            MCap_flag_B = 1 if Market_Cap_B<MCap_90_i else 0
            
            sum_A = lookback_flag_A + high_flag_A + MCap_flag_A
            sum_B = lookback_flag_B + high_flag_B + MCap_flag_B
            
            if (sum_A+sum_B) == 6:
                returns_series[j] = returns.loc[returns.index[j+1],top_x_i].mean(axis=0)
            elif sum_A==3:
                returns_series[j] = returns.loc[returns.index[j+1],top_x_i[0]]
            elif sum_B==3:
                returns_series[j] = returns.loc[returns.index[j+1],top_x_i[1]]
            else:
                returns_series[j] = 0

            strategy_returns_long.iloc[j] = returns.loc[returns.index[j+1],top_x_i].values
    returns_series = pd.DataFrame(returns_series, columns=['returns'])
    return returns_series, strategy_returns_long

def run_strategy_4_1(Lookback = 38,Selection = "R+H",Holding_Period = 1,Number_of_Stocks = 2):
    Timeframe = "Weekly"
    
    lookback_returns = return_lookback(weekly_returns,Lookback)
    periods_ud = Periods_Up_Down(weekly_returns,Lookback)

    High_ratio,Low_ratio  = timeframe_high_low_ratio(price_data,Lookback,'weekly')
    High_ratio = High_ratio.resample('W').last().bfill()
    Low_ratio = Low_ratio.resample('W').last().bfill()
    
    rank_df = rank_fn(lookback_returns,"Descending")
    rank_df_3 = rank_fn(High_ratio,"Descending")

    if Selection == 'R':
        top_x, _ = select_top_x_bottom_x(rank_df,Number_of_Stocks)
    if Selection == 'R+U':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2,Number_of_Stocks)
    if Selection == 'R+H':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_3,Number_of_Stocks)
    if Selection == 'R+U+H':
        top_x, _ = select_top_x_bottom_x(rank_df+rank_df_2+rank_df_3,Number_of_Stocks)
        
    returns_alloc, strategy_returns_all = trade_setup_4_1(weekly_returns, top_x, lookback_returns, High_ratio, 
                                                          Market_Cap, MCap_90, Holding_Period)
    returns_alloc = returns_alloc[(Lookback-1):-1]
    strategy_returns_all = strategy_returns_all.iloc[(Lookback-1):-1]
    
    strategy_df = pd.DataFrame(columns = ['Lookback','Rank','Holding Period','Number of stocks',
                                     'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
                                     'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])
    mean_return,cagr,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc,Timeframe)
    strategy_df = strategy_df.append([{'Lookback' : Lookback,
                                   'Rank' : Selection,
                                   'Holding Period': Holding_Period,
                                   'Number of stocks' : Number_of_Stocks,
                                   'CAGR' : np.round(cagr*100),
                                   'Mean Returns' : np.round(mean_return*100),
                                   'Standard Deviation' : np.round(st_dev*100),
                                   'Sharpe' : np.round(sharpe_ratio,2),
                                   'Sortino' : np.round(sortino_ratio,2),
                                   'Max Drawdown' : np.round(max_drawdown*100),
                                   'Calmar' : np.round(calmar_ratio,2),
                                   'Success Rate' : np.round(success_rate*100),
                                   'Average Up' : np.round(average_up*100),
                                   'Average Down' : np.round(average_down*100)}])
    
    print(strategy_df)
    
    return returns_alloc, lookback_returns, High_ratio, top_x, strategy_returns_all


# In[112]:


strategy_returns, lookback_returns, High_ratio, top_x, strategy_returns_all = run_strategy_4_1(38,'R+H',1,2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




