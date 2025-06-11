from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


app = Flask(__name__)
IMG_FOLDER = os.path.join('static', 'img')
os.makedirs(IMG_FOLDER, exist_ok=True)

# 載入已訓練模型
MODEL_PATH = os.path.join('models', 'a2c_2330.TW.zip')
model = A2C.load(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    # 1. 取得使用者輸入
    start_date       = request.form['start_date']
    end_date         = request.form['end_date']
    ticker           = request.form['ticker']
    init_amount      = float(request.form['init_amount'])
    transaction_cost = float(request.form['transaction_cost'])
    try:
        user_hmax = int(request.form['hmax'])
    except:
        user_hmax = 100
    hmax = max(1, min(user_hmax, 500))

    # 2. 抓資料
    df = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=[ticker]
    ).fetch_data()

    # 3. 特徵工程
    tech_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        tech_indicator_list=tech_list
    )
    df = fe.preprocess_data(df)

    # 4. 環境參數
    stock_dim   = 1
    state_space = 1 + len(tech_list) + 2 * stock_dim
    env_kwargs  = {
        'stock_dim': stock_dim,
        'hmax': hmax,
        'initial_amount': init_amount,
        'num_stock_shares': [0]*stock_dim,
        'buy_cost_pct': [transaction_cost]*stock_dim,
        'sell_cost_pct': [transaction_cost]*stock_dim,
        'reward_scaling': 1e-4,
        'state_space': state_space,
        'action_space': stock_dim,
        'tech_indicator_list': tech_list,
        'make_plots': False,
        'print_verbosity': 1
    }

    # 5. 建立環境並 reset
    env = StockTradingEnv(df=df, **env_kwargs)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = np.array(obs)

    # 6. 回測
    history = []
    done = False
    while not done:
        obs_batch = obs.reshape(1, -1)
        action_batch, _ = model.predict(obs_batch, deterministic=True)
        action = action_batch[0]
        step = env.step(action)
        if len(step) == 5:
            obs, reward, term, trunc, info = step
            done = term or trunc
        else:
            obs, reward, done, info = step
        history.append(env.asset_memory[-1])
        obs = np.array(obs)

    history_arr = np.array(history)
    daily_ret = history_arr[1:] / history_arr[:-1] - 1
    cumulative_return = history_arr[-1] / history_arr[0] - 1
    annual_return = (1 + cumulative_return) ** (252/len(daily_ret)) - 1
    annual_volatility = np.std(daily_ret, ddof=0) * np.sqrt(252)
    sharpe_ratio = np.mean(daily_ret)/np.std(daily_ret, ddof=0)*np.sqrt(252)
    downside = daily_ret[daily_ret<0]
    downside_std = np.std(downside, ddof=0) if len(downside)>0 else np.nan
    sortino_ratio = np.mean(daily_ret)/downside_std*np.sqrt(252) if downside_std>0 else np.nan
    running_max = np.maximum.accumulate(history_arr)
    drawdowns = history_arr/running_max - 1
    max_dd = drawdowns.min()
    calmar_ratio = annual_return/abs(max_dd) if max_dd<0 else np.nan
    var_95 = -np.percentile(daily_ret,5)

    # 7. 圖表
    # 資產淨值走勢
    plt.figure(figsize=(8,4))
    plt.plot(history_arr)
    plt.title('Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Total Asset')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER,'equity_curve.png'))
    plt.close()
    # 回撤曲線
    plt.figure(figsize=(8,4))
    plt.plot(drawdowns)
    plt.title('Drawdown Curve')
    plt.xlabel('Step')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER,'drawdown_curve.png'))
    plt.close()
    # 收益分佈直方圖
    plt.figure(figsize=(8,4))
    plt.hist(daily_ret, bins=50)
    plt.title('Return Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER,'returns_hist.png'))
    plt.close()

    # 8. 回傳
    return render_template('result.html', ticker=ticker, start_date=start_date, end_date=end_date,
        init_amount=init_amount, transaction_cost=transaction_cost, hmax=hmax,
        final_asset=history_arr[-1], cumulative_return=cumulative_return*100,
        annual_return=annual_return*100, annual_volatility=annual_volatility*100,
        sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio, calmar_ratio=calmar_ratio,
        max_drawdown=max_dd*100, var_95=var_95*100,
        equity_curve='equity_curve.png', drawdown_curve='drawdown_curve.png', returns_hist='returns_hist.png')

@app.route('/static/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(IMG_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
