import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 加在最前面，避免使用 tkinter
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import yfinance as yf
import mplfinance as mpf

def fetch_stock_codes(url):
    #"""從 TWSE 取得股票代號清單"""
    response = requests.get(url)
    response.encoding = 'big5'  # TWSE 網頁使用 big5 編碼
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'h4'})
    stock_codes = []
    for row in table.find_all('tr')[1:]:  # 跳過表頭
        cells = row.find_all('td')
        if len(cells) < 1:
            continue
        code_name = cells[0].text.strip()
        if len(code_name) > 4:
            code = code_name[:4]
            try:
                code_int = int(code)
                if 1101 <= code_int <= 9958:
                    stock_codes.append(f"{code}.TW")
            except ValueError:
                continue
    return stock_codes

def get_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def calculate_rma(series, period):
    #"""計算 RMA (類似 TradingView 的 ta.rma)"""
    rma = series.ewm(alpha=1/period, adjust=False).mean()
    return rma

def calculate_rsi(df, period=14):
    #"""使用 RMA 計算 RSI，模擬 TradingView 公式"""
    delta = df['Close'].diff(1)

    # 計算上漲與下跌值
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # 使用 RMA 計算平均漲跌幅
    avg_gain = calculate_rma(pd.Series(gain), period)
    avg_loss = calculate_rma(pd.Series(loss), period)
    
    rs = (avg_gain / avg_loss).round(2)
    df['RSI'] = (100 - (100 / (1 + pd.Series(rs.values, index=df.index))))

    df.dropna(inplace=True)
    return df
    

def calculate_rsi_based_ma(df, ma_length=14):
    #"""基於 RSI 計算移動平均線 (RSI-based MA)"""
    df['RSI_MA'] = df['RSI'].rolling(window=ma_length, min_periods=1).mean().round(2)
    return df

def get_last_trading_day():
    #"""返回最近的交易日（如果今天是假日）"""
    today = pd.Timestamp('today', tz="Asia/Taipei")
    if today.weekday() == 5:
        return today - pd.Timedelta(days=1)
    elif today.weekday() == 6:
        return today - pd.Timedelta(days=2)
    else:
        return today


def plot_stock_data(df, ticker, save_path=None):
    df = df.copy()
    
    # 確保 df 包含 datetime index 與必要欄位
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'RSI_MA', 'SMA_100']]
    df.index.name = 'Date'
    
    # 建立技術指標附圖 (RSI)
    apds = [
        mpf.make_addplot(df['SMA_100'], color='green'),         # 主圖上的 SMA
        mpf.make_addplot(df['RSI'], panel=1, color='red'),      # RSI 副圖
        mpf.make_addplot(df['RSI_MA'], panel=1, color='orange') # RSI_MA 副圖
    ]
    
    # 設定圖表樣式
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.sans-serif': ['Microsoft JhengHei']})

    # 畫出圖（儲存或顯示）
    if save_path:
        mpf.plot(df, type='candle', addplot=apds, style=style,
         title=f"{ticker} 技術圖",
         ylabel='價格',               # 主圖（K線）左側標籤
         ylabel_lower='RSI',         # 副圖（RSI）左側標籤
         volume=True,
         panel_ratios=(3, 1),
         savefig=save_path if save_path else None)
    else:
        mpf.plot(df, type='candle', addplot=apds, style=style,
         title=f"{ticker} 技術圖",
         ylabel='價格',               # 主圖（K線）左側標籤
         ylabel_lower='RSI',         # 副圖（RSI）左側標籤
         volume=True,
         panel_ratios=(3, 1))
         
        
def calculate_sma(df, period=100):
    #"""計算簡單移動平均線"""
    df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
    return df

def find_today_rsi_screening_condition(df):
    #"""篩選符合條件的股票"""
    df['RSI_Lower_Today'] = df['RSI'] < df['RSI'].shift(1)
    df['RSI_MA_Above_Today'] = df['RSI_MA'] > df['RSI_MA'].shift(1)
    df['RSI_MA_Below_55'] = df['RSI_MA'] < 55
    df['Above_100_SMA'] = df['Close'] > df['SMA_100']
    df['Volume_Above_50k'] = df['Volume'] > 50000

    valid_mask = df['RSI_Lower_Today'] & df['RSI_MA_Above_Today'] & df['RSI_MA_Below_55'] & df['Above_100_SMA'] & df['Volume_Above_50k']
    last_trading_day = get_last_trading_day()
    today_mask = df.index.date == last_trading_day.date()

    if today_mask.any():
        return df[today_mask & valid_mask]
    
    return pd.DataFrame(columns=['Close', 'RSI', 'RSI_MA', 'RSI_Lower_Today', 'RSI_MA_Above_Today', 'RSI_Diff_Within_3', 'RSI_MA_Below_55'])

def find_today_bearish_rebound_condition(df):
    """
    篩選 RSI 下降後 RSI_MA 反彈，且 RSI_MA 仍低於 31 的空頭反彈訊號
    """
    df['RSI_MS_Lower'] = df['RSI_MA'].shift(2) >= df['RSI_MA'].shift(1)
    df['RSI_MA_Above_Today'] = df['RSI_MA'] >= df['RSI_MA'].shift(1)
    df['RSI_MA_Below_31'] = df['RSI_MA'] <= 31
    df['Volume_Above_50k'] = df['Volume'] >= 300000

    valid_mask = (df['RSI_MS_Lower'] &
                  df['RSI_MA_Above_Today'] &
                  df['RSI_MA_Below_31'] &
                  df['Volume_Above_50k'])

    last_trading_day = get_last_trading_day()
    today_mask = df.index.date == last_trading_day.date()

    if today_mask.any():
        return df[today_mask & valid_mask]
    return pd.DataFrame(columns=df.columns)
