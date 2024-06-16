import pandas as pd
import numpy as np
import yfinance as yf
import datetime
# import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from time import sleep

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width
pd.set_option('display.expand_frame_repr', False)  # Do not wrap in repr


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0, :] = 0
    return returns

def daily_weekly_rollup(daily_returns, weekday):
    if weekday == 4 or weekday == 5 or weekday == 6:
        weekly_returns = (1 + daily_returns).resample('W').prod() - 1
    elif weekday == 0:
        weekly_returns = (1 + daily_returns).resample('W-MON').prod() - 1
    elif weekday == 1:
        weekly_returns = (1 + daily_returns).resample('W-TUE').prod() - 1
    elif weekday == 2:
        weekly_returns = (1 + daily_returns).resample('W-WED').prod() - 1
    elif weekday == 3:
        weekly_returns = (1 + daily_returns).resample('W-THU').prod() - 1
    return weekly_returns

def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

def weekly_monthly_rollup(weekly_returns):
    monthly_returns = (1 + weekly_returns).resample('M').prod() - 1
    return monthly_returns

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

def custom_allocator(trade_selection,capital,custom_allocation):
    trade_names = allocation(trade_selection, capital)
    trade_names['Num_Shares'] = custom_allocation
    trade_names['Allocation'] = round(trade_names['Num_Shares'] * trade_names['Price'],2)
    trade_names['Total_Capital'] = round(sum(trade_names['Allocation']),2)
    return trade_names

def run_strategy(today_date = '', use_stored = True):
    nifty500_scrips = pd.read_csv('ind_nifty500list.csv')
    nifty500_scrips['Scrips'] = nifty500_scrips['Symbol'] + ".NS"

    if today_date == '':
        today_date = datetime.datetime(2024, 6, 14)
    # 0 = Monday to 6 = Sunday, 4 = Friday
    weekday = today_date.weekday()

    if weekday == 4 or weekday == 5 or weekday == 6:
        days_to_subtract = (weekday + 3) % 7  # for Friday
    elif weekday == 0:
        days_to_subtract = (weekday + 7) % 7  #for Monday
    elif weekday == 1:
        days_to_subtract = (weekday + 6) % 7  #for Tuesday
    elif weekday == 2:
        days_to_subtract = (weekday + 5) % 7  #for Wednesday
    elif weekday == 3:
        days_to_subtract = (weekday + 4) % 7  #for Thursday

    last_weekday = today_date - datetime.timedelta(days=days_to_subtract)
    start_date = last_weekday - datetime.timedelta(weeks=52)

    print("start date = ", start_date)
    print("last weekday = ", last_weekday)

    price_data = yf.download(nifty500_scrips['Scrips'].to_list(), start=start_date,
                             end=last_weekday + datetime.timedelta(days=1))['Adj Close']

    print('Last Date:', price_data.index[-1])

    columns_to_drop = price_data.columns[(price_data.isna().sum() / price_data.shape[0]) > .2].to_list()
    price_data = price_data.drop(columns=columns_to_drop)
    print('Number of stocks analyzed:', len(price_data.columns))

    daily_returns = convert_prices_to_returns(price_data)
    daily_returns[daily_returns > 0.2] = 0
    weekly_returns = daily_weekly_rollup(daily_returns, weekday)
    print('Number of weeks:',weekly_returns.shape[0])

    lookback_returns = return_lookback(weekly_returns, 38)
    High_ratio = timeframe_high_ratio(price_data)

    rank_lookback = lookback_returns.rank(ascending=False)
    rank_highratio = High_ratio.rank(ascending=False)

    if use_stored==True:
        nifty500_scrips = pd.read_parquet('market_cap.parquet')
    else:
        nifty500_scrips['MarketCap'] = 0

        for i in range(nifty500_scrips.shape[0]):
            print('Scraping market cap data: ',i, " / ", nifty500_scrips.shape[0])
            url = 'https://www.screener.in/company/' + nifty500_scrips['Symbol'].iloc[i] + '/'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            nifty500_scrips['MarketCap'].iloc[i] = float(re.findall(r'\d+\.*\d*',
                                                                    text[text.find("Market Cap"):(
                                                                                text.find("Market Cap") + 60)].replace(",",
                                                                                                                       ""))[
                                                             0])
            sleep(1)

        nifty500_scrips.to_parquet('market_cap.parquet')

    Market_Cap = nifty500_scrips.copy()
    Market_Cap = Market_Cap[Market_Cap['Scrips'].isin(lookback_returns.index.to_list())]
    Market_Cap = Market_Cap[['Scrips', 'MarketCap']]
    Market_Cap.reset_index(inplace=True)

    Market_Cap_limit = np.percentile(Market_Cap['MarketCap'], 90)

    lookback_limit_A = 0.9
    lookback_limit_B = 0.71

    highratio_limit_A = 0.976
    highratio_limit_B = 0.962

    number_of_stocks = 4
    total_rank = rank_lookback + rank_highratio
    top_x = total_rank.nsmallest(number_of_stocks).index.tolist()

    top_lookback = rank_lookback.nsmallest(2).index.tolist()
    top_highratio = rank_highratio.nsmallest(2).index.tolist()

    print("Total Rank", top_x)
    print("Lookback Rank", top_lookback)
    print("High Ratio Rank", top_highratio)

    Trade_Df = pd.DataFrame(top_x, columns=['Stocks'])
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
        Trade_Df['Market_Cap'].iloc[i] = Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == Trade_Df['Stocks'].iloc[i]]

    Trade_Df['Lookback_Flag'].iloc[0] = 1 if Trade_Df['Lookback_Returns'].iloc[0] > lookback_limit_A else 0
    Trade_Df['Lookback_Flag'].iloc[1] = 1 if Trade_Df['Lookback_Returns'].iloc[1] > lookback_limit_B else 0

    Trade_Df['Highratio_Flag'].iloc[0] = 1 if Trade_Df['High_Ratio'].iloc[0] > highratio_limit_A else 0
    Trade_Df['Highratio_Flag'].iloc[1] = 1 if Trade_Df['High_Ratio'].iloc[1] > highratio_limit_B else 0

    Trade_Df['Market_Cap_Flag'].iloc[0] = 1 if Trade_Df['Market_Cap'].iloc[0] < Market_Cap_limit else 0
    Trade_Df['Market_Cap_Flag'].iloc[1] = 1 if Trade_Df['Market_Cap'].iloc[1] < Market_Cap_limit else 0

    lookback_dict_1 = {'Stocks': top_lookback[0], 'Lookback_Returns': lookback_returns[top_lookback[0]],
                       'Lookback_Rank': rank_lookback[top_lookback[0]], 'High_Ratio': High_ratio[top_lookback[0]],
                       'Highratio_Rank': rank_highratio[top_lookback[0]], 'Total_Rank': total_rank[top_lookback[0]],
                       'Market_Cap': Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == top_lookback[0]].iloc[0],
                       'Lookback_Flag': 0, 'Highratio_Flag': 0, 'Market_Cap_Flag': 0}

    lookback_dict_2 = {'Stocks': top_lookback[1], 'Lookback_Returns': lookback_returns[top_lookback[1]],
                       'Lookback_Rank': rank_lookback[top_lookback[1]], 'High_Ratio': High_ratio[top_lookback[1]],
                       'Highratio_Rank': rank_highratio[top_lookback[1]], 'Total_Rank': total_rank[top_lookback[1]],
                       'Market_Cap': Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == top_lookback[1]].iloc[0],
                       'Lookback_Flag': 0, 'Highratio_Flag': 0, 'Market_Cap_Flag': 0}

    highratio_dict_1 = {'Stocks': top_highratio[0], 'Lookback_Returns': lookback_returns[top_highratio[0]],
                        'Lookback_Rank': rank_lookback[top_highratio[0]], 'High_Ratio': High_ratio[top_highratio[0]],
                        'Highratio_Rank': rank_highratio[top_highratio[0]], 'Total_Rank': total_rank[top_highratio[0]],
                        'Market_Cap': Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == top_highratio[0]].iloc[0],
                        'Lookback_Flag': 0, 'Highratio_Flag': 0, 'Market_Cap_Flag': 0}

    highratio_dict_2 = {'Stocks': top_highratio[1], 'Lookback_Returns': lookback_returns[top_highratio[1]],
                        'Lookback_Rank': rank_lookback[top_highratio[1]], 'High_Ratio': High_ratio[top_highratio[1]],
                        'Highratio_Rank': rank_highratio[top_highratio[1]], 'Total_Rank': total_rank[top_highratio[1]],
                        'Market_Cap': Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == top_highratio[1]].iloc[0],
                        'Lookback_Flag': 0, 'Highratio_Flag': 0, 'Market_Cap_Flag': 0}

    Trade_Df = Trade_Df._append(lookback_dict_1, ignore_index=True)
    Trade_Df = Trade_Df._append(lookback_dict_2, ignore_index=True)
    Trade_Df = Trade_Df._append(highratio_dict_1, ignore_index=True)
    Trade_Df = Trade_Df._append(highratio_dict_2, ignore_index=True)

    Trade_Df_Strategy_1 = Trade_Df.copy()

    ma_50 = price_data.rolling(window=50).mean()
    perc_distance_ma_50 = abs(price_data.iloc[-1] / ma_50.iloc[-1] - 1)
    rank_50ma = perc_distance_ma_50.rank(ascending=True)

    number_of_stocks = 8
    total_rank = rank_lookback + rank_highratio + rank_50ma
    top_x = total_rank.nsmallest(number_of_stocks).index.tolist()

    Trade_Df = pd.DataFrame(top_x, columns=['Stocks'])
    Trade_Df['Lookback_Returns'] = 0
    Trade_Df['Lookback_Rank'] = 0
    Trade_Df['High_Ratio'] = 0
    Trade_Df['Highratio_Rank'] = 0
    Trade_Df['Distance_MA50'] = 0
    Trade_Df['Distance_MA50_Rank'] = 0
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
        Trade_Df['Distance_MA50'].iloc[i] = perc_distance_ma_50[Trade_Df['Stocks'].iloc[i]]
        Trade_Df['Distance_MA50_Rank'].iloc[i] = rank_50ma[Trade_Df['Stocks'].iloc[i]]
        Trade_Df['Total_Rank'].iloc[i] = total_rank[Trade_Df['Stocks'].iloc[i]]
        Trade_Df['Market_Cap'].iloc[i] = Market_Cap['MarketCap'].loc[Market_Cap['Scrips'] == Trade_Df['Stocks'].iloc[i]]

    Trade_Df['Lookback_Flag'].iloc[0] = 1 if Trade_Df['Lookback_Returns'].iloc[0] > lookback_limit_A else 0
    Trade_Df['Lookback_Flag'].iloc[1] = 1 if Trade_Df['Lookback_Returns'].iloc[1] > lookback_limit_B else 0

    Trade_Df['Highratio_Flag'].iloc[0] = 1 if Trade_Df['High_Ratio'].iloc[0] > highratio_limit_A else 0
    Trade_Df['Highratio_Flag'].iloc[1] = 1 if Trade_Df['High_Ratio'].iloc[1] > highratio_limit_B else 0

    Trade_Df['Market_Cap_Flag'].iloc[0] = 1 if Trade_Df['Market_Cap'].iloc[0] < Market_Cap_limit else 0
    Trade_Df['Market_Cap_Flag'].iloc[1] = 1 if Trade_Df['Market_Cap'].iloc[1] < Market_Cap_limit else 0

    Trade_Df_Strategy_2 = Trade_Df.copy()

    return Trade_Df_Strategy_1, Trade_Df_Strategy_2

if __name__ == '__main__':
    # today_date = datetime.datetime(2024, 6, 11)
    Trade_Df_Strategy_1, Trade_Df_Strategy_2 = run_strategy(today_date = '',use_stored=True)

    print('Strategy 1')
    print(Trade_Df_Strategy_1)
    print('Strategy 2')
    print(Trade_Df_Strategy_2)
