#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from time import sleep


# In[2]:


nifty500_scrips = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/Systematic Strategies/NIFTY Strategy/Nifty500_List/ind_nifty500list.csv')
nifty500_scrips['Scrips'] = nifty500_scrips['Symbol'] + ".NS"


# In[3]:


today_date = datetime.datetime(2024,5,9)
#0 = Monday to 6 = Sunday, 4 = Friday
weekday = today_date.weekday()
days_to_subtract = (weekday + 3) % 7
last_friday = today_date - datetime.timedelta(days=days_to_subtract)
start_date = last_friday - datetime.timedelta(weeks=52)

print("start date = ", start_date)
print("last friday = ", last_friday)


# In[4]:


price_data = yf.download(nifty500_scrips['Scrips'].to_list(), start=start_date, end=last_friday)['Adj Close']


# In[5]:


columns_to_drop = price_data.columns[price_data.isna().sum()/price_data.shape[0]>.1].to_list()
price_data = price_data.drop(columns=columns_to_drop)


# In[6]:


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


# In[7]:


daily_returns = convert_prices_to_returns(price_data)
daily_returns[daily_returns > 0.2] = 0
weekly_returns = daily_weekly_rollup(daily_returns)


# In[24]:


def return_lookback(returns,lookback_period = 38):
    lookback_returns = ((1 + returns).rolling(window=lookback_period).apply(lambda x: x.prod()) - 1).iloc[-1]
    return lookback_returns

def timeframe_high_ratio(prices):
    high52_ratio = prices.iloc[-1]/prices.max()
    return high52_ratio

def rank_fn(indicator,order=True):
    asc_flag = True if order=='Ascending' else False
    rank_df = indicator.rank(axis=1, ascending = asc_flag)
    return rank_df

def select_top_x(rank,x = 2):
    top_x = rank.apply(lambda row: row.nsmallest(x).index.tolist(), axis=1)
    return top_x


# In[9]:


lookback_returns = return_lookback(weekly_returns,38)
High_ratio = timeframe_high_ratio(price_data)


# In[10]:


rank_lookback = lookback_returns.rank(ascending = False)
rank_highratio = High_ratio.rank(ascending = False)


# In[11]:


nifty500_scrips['MarketCap'] = 0

for i in range(nifty500_scrips.shape[0]):
    print(i, " / ", nifty500_scrips.shape[0])
    url = 'https://www.screener.in/company/' + nifty500_scrips['Symbol'].iloc[i] + '/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    nifty500_scrips['MarketCap'].iloc[i] = float(re.findall(r'\d+\.*\d*', 
                                text[text.find("Market Cap"):(text.find("Market Cap")+60)].replace(",", ""))[0])
    sleep(1)


# In[12]:


Market_Cap = nifty500_scrips.copy()
Market_Cap = Market_Cap[Market_Cap['Scrips'].isin(lookback_returns.index.to_list())]
Market_Cap = Market_Cap[['Scrips','MarketCap']]
Market_Cap.reset_index(inplace=True)

Market_Cap_limit = np.percentile(Market_Cap['MarketCap'],90) 

lookback_limit_A = 0.9
lookback_limit_B = 0.71

highratio_limit_A = 0.976
highratio_limit_B = 0.962


# In[29]:


number_of_stocks = 5
total_rank = rank_lookback + rank_highratio
top_x = total_rank.nsmallest(number_of_stocks).index.tolist()
print(top_x)


# In[32]:


Trade_Df = pd.DataFrame(top_x, columns = ['Stocks'])
Trade_Df['Lookback_Returns'] = 0
Trade_Df['Lookback_Rank'] = 0
Trade_Df['High_Ratio'] = 0
Trade_Df['Highratio_Rank'] = 0
Trade_Df['Total_Rank'] = 0
Trade_Df['Market_Cap'] = 0
Trade_Df['Lookback_Flag'] = 0
Trade_Df['Highratio_Flag'] = 0
Trade_Df['Market_Cap_Flag'] = 0

for i in range(Trade_Df.shape[0]):
    Trade_Df['Lookback_Returns'].iloc[i] = lookback_returns[Trade_Df['Stocks'].iloc[i]]
    Trade_Df['Lookback_Rank'].iloc[i] = rank_lookback[Trade_Df['Stocks'].iloc[i]]
    Trade_Df['High_Ratio'].iloc[i] = High_ratio[Trade_Df['Stocks'].iloc[i]]
    Trade_Df['Highratio_Rank'].iloc[i] = rank_highratio[Trade_Df['Stocks'].iloc[i]]
    Trade_Df['Total_Rank'].iloc[i] = total_rank[Trade_Df['Stocks'].iloc[i]]
    Trade_Df['Market_Cap'].iloc[i] = Market_Cap['MarketCap'].loc[Market_Cap['Scrips']==Trade_Df['Stocks'].iloc[i]]

Trade_Df['Lookback_Flag'].iloc[0] = 1 if Trade_Df['Lookback_Returns'].iloc[0]>lookback_limit_A else 0
Trade_Df['Lookback_Flag'].iloc[1] = 1 if Trade_Df['Lookback_Returns'].iloc[1]>lookback_limit_B else 0

Trade_Df['Highratio_Flag'].iloc[0] = 1 if Trade_Df['High_Ratio'].iloc[0]>highratio_limit_A else 0
Trade_Df['Highratio_Flag'].iloc[1] = 1 if Trade_Df['High_Ratio'].iloc[1]>highratio_limit_B else 0

Trade_Df['Market_Cap_Flag'].iloc[0] = 1 if Trade_Df['Market_Cap'].iloc[0]<Market_Cap_limit else 0
Trade_Df['Market_Cap_Flag'].iloc[1] = 1 if Trade_Df['Market_Cap'].iloc[1]<Market_Cap_limit else 0


# In[80]:


top_lookback = rank_lookback.nsmallest(2).index.tolist()
top_highratio = rank_highratio.nsmallest(2).index.tolist()

lookback_dict_1 = {'Stocks' : top_lookback[0], 'Lookback_Returns' : lookback_returns[top_lookback[0]],
                 'Lookback_Rank' : rank_lookback[top_lookback[0]], 'High_Ratio' : High_ratio[top_lookback[0]],
                 'Highratio_Rank' : rank_highratio[top_lookback[0]], 'Total_Rank' : total_rank[top_lookback[0]],
                 'Market_Cap' : Market_Cap['MarketCap'].loc[Market_Cap['Scrips']==top_lookback[0]].iloc[0],
                 'Lookback_Flag' : 0, 'Highratio_Flag' : 0, 'Market_Cap_Flag' : 0}

lookback_dict_2 = {'Stocks' : top_lookback[1], 'Lookback_Returns' : lookback_returns[top_lookback[1]],
                 'Lookback_Rank' : rank_lookback[top_lookback[1]], 'High_Ratio' : High_ratio[top_lookback[1]],
                 'Highratio_Rank' : rank_highratio[top_lookback[1]], 'Total_Rank' : total_rank[top_lookback[1]],
                 'Market_Cap' : Market_Cap['MarketCap'].loc[Market_Cap['Scrips']==top_lookback[1]].iloc[0],
                 'Lookback_Flag' : 0, 'Highratio_Flag' : 0, 'Market_Cap_Flag' : 0}

highratio_dict_1 = {'Stocks' : top_highratio[0], 'Lookback_Returns' : lookback_returns[top_highratio[0]],
                 'Lookback_Rank' : rank_lookback[top_highratio[0]], 'High_Ratio' : High_ratio[top_highratio[0]],
                 'Highratio_Rank' : rank_highratio[top_highratio[0]], 'Total_Rank' : total_rank[top_highratio[0]],
                 'Market_Cap' : Market_Cap['MarketCap'].loc[Market_Cap['Scrips']==top_highratio[0]].iloc[0],
                 'Lookback_Flag' : 0, 'Highratio_Flag' : 0, 'Market_Cap_Flag' : 0}

highratio_dict_2 = {'Stocks' : top_highratio[1], 'Lookback_Returns' : lookback_returns[top_highratio[1]],
                 'Lookback_Rank' : rank_lookback[top_highratio[1]], 'High_Ratio' : High_ratio[top_highratio[1]],
                 'Highratio_Rank' : rank_highratio[top_highratio[1]], 'Total_Rank' : total_rank[top_highratio[1]],
                 'Market_Cap' : Market_Cap['MarketCap'].loc[Market_Cap['Scrips']==top_highratio[1]].iloc[0],
                 'Lookback_Flag' : 0, 'Highratio_Flag' : 0, 'Market_Cap_Flag' : 0}

Trade_Df = Trade_Df.append(lookback_dict_1, ignore_index=True)
Trade_Df = Trade_Df.append(lookback_dict_2, ignore_index=True)
Trade_Df = Trade_Df.append(highratio_dict_1, ignore_index=True)
Trade_Df = Trade_Df.append(highratio_dict_2, ignore_index=True)


# In[ ]:





# In[ ]:





# In[ ]:




