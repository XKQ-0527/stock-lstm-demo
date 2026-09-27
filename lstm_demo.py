# -*- coding: utf-8 -*-
"""
LSTM 股价预测入门 demo
======================
任务：用过去 WINDOW 天的收盘价，预测下一个交易日的收盘价。
对照：朴素基线 "明天价格 = 今天价格"。

运行方式（二选一）：
  1. 终端：conda activate quant && python lstm_demo.py
  2. VS Code：打开本文件，点击每个 "# %%" 上方的 "Run Cell"，一段一段运行（推荐，便于观察）

依赖：akshare pandas numpy matplotlib torch
"""

# %% 1. 导入需要的库 ----------------------------------------------------------
import numpy as np                 # 数值计算（相当于 Java 里的数组 + 数学库）
import pandas as pd                # 表格数据处理（相当于一个内存里的 Excel）
import matplotlib.pyplot as plt    # 画图
import torch                       # 深度学习框架 PyTorch
import torch.nn as nn              # 神经网络模块
import akshare as ak               # 免费的 A 股数据接口

# 让 Mac 上的 matplotlib 能显示中文
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子：保证每次运行结果一致，方便对比
torch.manual_seed(42)
np.random.seed(42)

# ---- 可以自己修改的参数 ----
SYMBOL = "600519"          # 股票代码：600519 = 贵州茅台
START, END = "20190101", "20251231"
WINDOW = 30                # 用过去 30 天预测下一天
TRAIN_RATIO = 0.8          # 前 80% 的时间用来训练，后 20% 用来测试
EPOCHS = 60                # 训练轮数
HIDDEN = 64                # LSTM 隐藏层大小


# %% 2. 获取数据 ---------------------------------------------------------------
def load_close_prices(symbol: str) -> pd.DataFrame:
    """返回只有 date、close 两列的表格，按日期升序排列。"""
    try:
        # 数据源 1：东方财富（前复权）
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=START, end_date=END, adjust="qfq")
        df = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})
    except Exception as e:
        print("东方财富接口失败，改用新浪接口：", e)
        # 数据源 2：新浪（代码前要加 sh / sz）
        prefix = "sh" if symbol.startswith("6") else "sz"
        df = ak.stock_zh_a_daily(symbol=prefix + symbol, start_date=START,
                                 end_date=END, adjust="qfq")
        df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


df = load_close_prices(SYMBOL)
print(df.head())                    # 看前 5 行
print("共", len(df), "个交易日")

# 画一下原始收盘价
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["close"])
plt.title(f"{SYMBOL} 收盘价（前复权）")
plt.tight_layout()
plt.savefig("price.png", dpi=120)
plt.show()


# %% 3. 数据预处理：划分、归一化、构造滑动窗口 ---------------------------------
prices = df["close"].values.astype(np.float32)        # 转成 numpy 数组
split = int(len(prices) * TRAIN_RATIO)                 # 训练/测试的分界点

# 归一化：把价格缩放到 0~1 之间，神经网络更好训练。
# 关键：最小值、最大值只能用"训练集"来算！
# 如果用全部数据，就等于偷看了未来的价格（数据泄露）。
p_min, p_max = prices[:split].min(), prices[:split].max()
scaled = (prices - p_min) / (p_max - p_min)


def make_windows(series: np.ndarray, window: int):
    """把一条序列切成 (过去 window 天 -> 下一天) 的样本对。"""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])   # 第 i-window ~ i-1 天
        y.append(series[i])              # 第 i 天（要预测的目标）
    return np.array(X), np.array(y)


X, y = make_windows(scaled, WINDOW)
# 第 k 个样本的目标是原序列的第 k+WINDOW 天，
# 所以"目标日期在训练期内"的样本是前 split-WINDOW 个。
n_train = split - WINDOW
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# 转成 PyTorch 需要的张量形状：(样本数, 时间步, 特征数=1)
to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
X_train_t = to_tensor(X_train).unsqueeze(-1)
X_test_t = to_tensor(X_test).unsqueeze(-1)
y_train_t = to_tensor(y_train).unsqueeze(-1)

print("训练样本:", X_train_t.shape, " 测试样本:", X_test_t.shape)


# %% 4. 定义 LSTM 模型 ---------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size=HIDDEN, num_layers=2):
        super().__init__()
        # LSTM：逐天读入价格，把"记忆"存在隐藏状态里
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # 全连接层：把最后一天的隐藏状态映射成 1 个数（预测价格）
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out 形状: (batch, WINDOW, hidden)
        return self.fc(out[:, -1, :])  # 只取最后一个时间步


model = LSTMPredictor()
loss_fn = nn.MSELoss()                                   # 均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# %% 5. 训练 -------------------------------------------------------------------
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, y_train_t),
    batch_size=64, shuffle=True)   # 窗口内部的时间顺序不变，打乱的只是样本的先后，没问题

losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()           # 1. 清空上一步的梯度
        loss = loss_fn(model(xb), yb)   # 2. 前向计算 + 求损失
        loss.backward()                 # 3. 反向传播求梯度
        optimizer.step()                # 4. 更新参数
        total += loss.item() * len(xb)
    losses.append(total / len(X_train_t))
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  train MSE = {losses[-1]:.6f}")

plt.figure(figsize=(6, 3))
plt.plot(losses)
plt.title("训练损失曲线")
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig("loss.png", dpi=120)
plt.show()


# %% 6. 在测试集上评估，并与朴素基线对比 ---------------------------------------
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test_t).squeeze(-1).numpy()

# 反归一化回真实价格
inverse = lambda a: a * (p_max - p_min) + p_min
y_true = inverse(y_test)
y_lstm = inverse(pred_scaled)
y_naive = inverse(X_test[:, -1])      # 朴素基线：直接用"今天"的价格当"明天"的预测


def report(name, pred):
    mae = np.mean(np.abs(pred - y_true))
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    # 方向准确率：预测的涨跌方向（相对今天）是否与真实方向一致
    today = inverse(X_test[:, -1])
    direction = np.mean(np.sign(pred - today) == np.sign(y_true - today))
    print(f"{name:<10s} MAE = {mae:8.2f}   RMSE = {rmse:8.2f}   方向准确率 = {direction:.1%}")
    return mae, rmse


print("\n==== 测试集结果 ====")
report("LSTM", y_lstm)
report("朴素基线", y_naive)
print("（朴素基线预测的变化量永远是 0，所以它的方向准确率没有意义）")

test_dates = df["date"].values[split:]
plt.figure(figsize=(11, 4))
plt.plot(test_dates, y_true, label="真实价格", linewidth=1.5)
plt.plot(test_dates, y_lstm, label="LSTM 预测", linewidth=1)
plt.plot(test_dates, y_naive, label="朴素基线（今天=明天）", linewidth=1, alpha=0.6)
plt.legend()
plt.title(f"{SYMBOL} 测试集：LSTM vs 朴素基线")
plt.tight_layout()
plt.savefig("result.png", dpi=120)
plt.show()

# %% 7. 思考题（写进 README / 邮件里的素材）------------------------------------
# 1. LSTM 是否明显优于"今天=明天"？为什么预测"价格"这么难？
# 2. 把 WINDOW 改成 10 或 60，结果如何变化？
# 3. 如果改成预测"涨跌幅"而不是"价格"，会怎样？（提示：价格序列高度自相关）
# 4. 除了收盘价，加入成交量、开盘价等特征会不会更好？
# 5. 对照论文①：Informer 相比 LSTM 的优势是什么？三日 K 线形态识别起了什么作用？
