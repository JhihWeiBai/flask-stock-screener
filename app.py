from flask import Flask, render_template
from stock_utils import (
    fetch_stock_codes,
    get_stock_data,
    calculate_rsi,
    calculate_rsi_based_ma,
    calculate_sma,
    find_today_rsi_screening_condition,
    find_today_bearish_rebound_condition,
    plot_stock_data  # 確保這個函式也在 stock_utils 裡
)
import os

app = Flask(__name__)   #建立一個網站主體（Flask 應用物件）

@app.route("/")         #定義網站首頁
def index():
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    stock_codes = fetch_stock_codes(url)
    print(f"取得 {len(stock_codes)} 檔股票資料。")

    rebound_stocks = []
    bearish_rebound_stocks = []

    for code in stock_codes:
        try:
            df = get_stock_data(code)
            if df.empty:
                continue
            df = calculate_rsi(df)
            df = calculate_rsi_based_ma(df)
            df = calculate_sma(df, period=100)

            # 回測 RSI 向上突破
            if not find_today_rsi_screening_condition(df).empty:
                save_path = f"static/charts/{code}.png"
                plot_stock_data(df, code, save_path)
                rebound_stocks.append(code)

            # 空頭反彈訊號
            if not find_today_bearish_rebound_condition(df).empty:
                save_path = f"static/charts/{code}.png"
                plot_stock_data(df, code, save_path)
                bearish_rebound_stocks.append(code)

        except Exception as e:
            print(f"{code} 發生錯誤：{e}")
            continue

    return render_template("index.html", 
                           stocks=rebound_stocks, 
                           bearish_stocks=bearish_rebound_stocks)

@app.route("/chart/<ticker>")
def chart(ticker):
    df = get_stock_data(ticker)
    df = calculate_rsi(df)
    df = calculate_rsi_based_ma(df)
    df = calculate_sma(df, period=100)

    # 使用已封裝好的函式產圖（若不存在則產生）
    save_path = f"static/charts/{ticker}.png"
    if not os.path.exists(save_path):
        plot_stock_data(df, ticker, save_path)

    return render_template("chart.html", ticker=ticker, img_file=save_path)

if __name__ == "__main__":
    print("*Flask is starting...")
    app.run(debug=True)             #啟動網站伺服器