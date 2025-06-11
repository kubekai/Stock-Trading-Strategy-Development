import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent


def compute_performance_metrics(df_value):
    """
    計算年化報酬、Sharpe ratio、最大回撤。
    假設 df_value 裡有 'account_value' 欄位。
    """
    df = df_value.copy().reset_index(drop=True)
    df["daily_return"] = df["account_value"].pct_change().fillna(0)

    total_days = len(df)
    cumulative_return = df["account_value"].iloc[-1] / df["account_value"].iloc[0] - 1
    annual_return = (1 + cumulative_return) ** (252 / total_days) - 1

    annual_volatility = df["daily_return"].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    roll_max = df["account_value"].cummax()
    drawdown = df["account_value"] / roll_max - 1
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }


if __name__ == "__main__":
    # ------------------------
    # 1. 下載 & 前處理
    # ------------------------
    ticker = "2330.TW"
    df = YahooDownloader(
        start_date="2018-01-01",
        end_date="2024-12-31",
        ticker_list=[ticker]
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "rsi", "cci", "dx"],
        use_turbulence=True,
        user_defined_feature=False
    )
    df_fe = fe.preprocess_data(df)
    df_fe.dropna(inplace=True)

    # ------------------------
    # 2. 切 train / val
    # ------------------------
    train_df = data_split(df_fe, "2018-01-01", "2023-01-01").reset_index(drop=True)
    val_df   = data_split(df_fe, "2023-01-02", "2024-12-31").reset_index(drop=True)

    # ------------------------
    # 3. 環境參數設定
    # ------------------------
    stock_dim        = train_df.tic.nunique()
    state_space      = 1 + 2 * stock_dim + len(fe.tech_indicator_list) * stock_dim
    action_space     = stock_dim
    num_stock_shares = [0] * stock_dim
    buy_cost_list    = [0.001425] * stock_dim
    sell_cost_list   = [0.001425] * stock_dim
    reward_scaling   = 1e-2
    turbulence_th    = train_df["turbulence"].quantile(0.95)

    # 建立向量化訓練環境
    train_env_sb3, _ = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=buy_cost_list,
        sell_cost_pct=sell_cost_list,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=fe.tech_indicator_list,
        turbulence_threshold=None
    ).get_sb_env()

    # ------------------------
    # 4. 多算法訓練 & 驗證
    # ------------------------
    algos = [
        {"name": "a2c", "timesteps": 5000000},
        {"name": "ppo", "timesteps": 5000000},
        {"name": "ddpg", "timesteps": 5000000},
        {"name": "td3", "timesteps": 5000000},
        {"name": "sac", "timesteps": 1_000_000},
    ]

    results = []
    eq_curves = {}
    failed_algos = []

    for algo in algos:
        print(f"Training {algo['name']}...")
        try:
            agent = DRLAgent(env=train_env_sb3)
            model = agent.get_model(algo["name"])
            trained = agent.train_model(
                model=model,
                tb_log_name=f"{algo['name']}_{ticker}",
                total_timesteps=algo["timesteps"]
            )
            model_path = f"{algo['name']}_{ticker}.zip"
            trained.save(model_path)

            val_env = StockTradingEnv(
                df=val_df,
                stock_dim=stock_dim,
                hmax=150,
                initial_amount=10000000,
                num_stock_shares=num_stock_shares,
                buy_cost_pct=buy_cost_list,
                sell_cost_pct=sell_cost_list,
                reward_scaling=reward_scaling,
                state_space=state_space,
                action_space=action_space,
                tech_indicator_list=fe.tech_indicator_list,
                turbulence_threshold=turbulence_th
            )
            df_val, _ = agent.DRL_prediction(
                model=trained,
                environment=val_env
            )
            perf = compute_performance_metrics(df_val)
            results.append({
                "algo": algo["name"],
                "model_path": model_path,
                **perf
            })
            eq_curves[algo["name"]] = df_val["account_value"].reset_index(drop=True)
            print(f"✔ {algo['name']}  Sharpe={perf['sharpe_ratio']:.2f}, "
                  f"AnnRet={perf['annual_return']:.2%}, MaxDD={perf['max_drawdown']:.2%}")
        except Exception as e:
            print(f"⚠ Skip {algo['name']} due to error: {e}")
            failed_algos.append(algo['name'])

    if failed_algos:
        print(f"Algorithms failed: {', '.join(failed_algos)}")
        # 移除失敗的算法，避免後續比較
        for f in failed_algos:
            eq_curves.pop(f, None)
        results = [r for r in results if r['algo'] not in failed_algos]

    if not results:
        print("No algorithm completed successfully. Please check environment and data.")
        exit(1)

    # ------------------------
    # 5. 圖表比較分析
    # ------------------------
    metrics_df = pd.DataFrame(results)

    # Equity Curve 比較
    plt.figure()
    for name, curve in eq_curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.legend(); plt.title("Equity Curve Comparison");
    plt.xlabel("Time Step"); plt.ylabel("Account Value");
    plt.tight_layout(); plt.savefig("equity_curve_comparison.png"); plt.close()

    # 指標長條圖
    for col, ylabel, fname in [
        ("sharpe_ratio", "Sharpe Ratio", "sharpe_ratio_comparison.png"),
        ("annual_return", "Annual Return", "annual_return_comparison.png"),
        ("max_drawdown", "Max Drawdown", "max_drawdown_comparison.png")
    ]:
        plt.figure();
        values = -metrics_df[col] if col=="max_drawdown" else metrics_df[col]
        plt.bar(metrics_df["algo"], values)
        plt.title(f"{ylabel} Comparison");
        plt.xlabel("Algorithm"); plt.ylabel(ylabel);
        plt.tight_layout(); plt.savefig(fname); plt.close()

    # Radar Chart
    import numpy as np
    labels = metrics_df["algo"].tolist()
    stats = np.vstack([
        metrics_df["annual_return"].values,
        metrics_df["sharpe_ratio"].values,
        -metrics_df["max_drawdown"].values
    ]).T
    angles = np.linspace(0, 2*np.pi, stats.shape[1], endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(); ax = fig.add_subplot(111, polar=True)
    for i, lbl in enumerate(labels):
        vals = stats[i].tolist(); vals += vals[:1]
        ax.plot(angles, vals, linewidth=1, label=lbl);
        ax.fill(angles, vals, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), ["AnnRet","Sharpe","MaxDD"])
    ax.set_title("Algorithm Performance Radar Chart"); ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.tight_layout(); plt.savefig("radar_comparison.png"); plt.close()
