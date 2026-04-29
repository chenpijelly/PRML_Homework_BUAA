# =========================================================
# 基于 LSTM / GRU / XGBoost 的 PM2.5 多变量时间序列预测
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =========================================================
# 1. 参数设置
# =========================================================

DATA_PATH = r"C:\Users\86137\PRML_Homework\LSTM-Multivariate_pollution.csv"

LOOK_BACK = 24
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# =========================================================
# 2. 数据读取与预处理
# =========================================================

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print("=" * 80)
    print("原始数据前5行：")
    print(df.head())
    print("原始数据维度：", df.shape)

    df.columns = [col.strip() for col in df.columns]

    # 兼容当前数据集列名
    df = df.rename(columns={
        "pollution": "pm2.5",
        "dew": "DEWP",
        "temp": "TEMP",
        "press": "PRES",
        "wnd_dir": "cbwd",
        "wnd_spd": "Iws",
        "snow": "Is",
        "rain": "Ir"
    })

    # 时间排序
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    elif all(col in df.columns for col in ["year", "month", "day", "hour"]):
        df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.sort_values("date").reset_index(drop=True)

    # 删除无关列
    if "No" in df.columns:
        df = df.drop(columns=["No"])

    # 风向编码
    if "cbwd" in df.columns:
        encoder = LabelEncoder()
        df["cbwd"] = encoder.fit_transform(df["cbwd"].astype(str))

    feature_cols = [
        "pm2.5",
        "DEWP",
        "TEMP",
        "PRES",
        "cbwd",
        "Iws",
        "Is",
        "Ir"
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少以下列：{missing_cols}")

    data = df[feature_cols].copy()
    data = data.replace("NA", np.nan)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.ffill().bfill()

    print("=" * 80)
    print("预处理后缺失值统计：")
    print(data.isnull().sum())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return data, scaled_data, scaler, feature_cols


# =========================================================
# 3. 构造时间序列样本
# =========================================================

def create_sequences(data_array, look_back=24, target_col=0):
    X, y = [], []

    for i in range(look_back, len(data_array)):
        X.append(data_array[i - look_back:i, :])
        y.append(data_array[i, target_col])

    return np.array(X), np.array(y)


# =========================================================
# 4. 时间顺序划分训练集和测试集
# =========================================================

def split_train_test(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test


# =========================================================
# 5. LSTM 模型
# =========================================================

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# =========================================================
# 6. GRU 模型
# =========================================================

def build_gru_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# =========================================================
# 7. XGBoost 模型
# =========================================================

def flatten_sequence_data(X):
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def build_xgboost_model():
    if not XGBOOST_AVAILABLE:
        return None

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED
    )

    return model


# =========================================================
# 8. 反归一化
# =========================================================

def inverse_transform_target(y_scaled, scaler, n_features, target_col=0):
    y_scaled = np.array(y_scaled).reshape(-1)

    temp = np.zeros((len(y_scaled), n_features))
    temp[:, target_col] = y_scaled

    inv = scaler.inverse_transform(temp)

    return inv[:, target_col]


# =========================================================
# 9. 评价指标
# =========================================================

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100


def pearson_corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape_value = smape(y_true, y_pred)
    corr = pearson_corr(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "SMAPE": smape_value,
        "Pearson": corr
    }


# =========================================================
# 10. 可视化函数
# =========================================================

def plot_raw_pm25(data):
    plt.figure(figsize=(12, 5))
    plt.plot(data["pm2.5"], label="PM2.5 / Pollution")
    plt.xlabel("Time Index")
    plt.ylabel("PM2.5")
    plt.title("Figure 1. Original PM2.5 Time Series")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_loss(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Figure 2. Training and Validation Loss of {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_curve(y_true, y_pred, model_name, num_points=300):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:num_points], label="True PM2.5", linewidth=2)
    plt.plot(y_pred[:num_points], label=f"{model_name} Predicted PM2.5")
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5")
    plt.title(f"Figure 3. True vs Predicted PM2.5 by {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_predictions(y_true, pred_dict, num_points=300):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:num_points], label="True PM2.5", linewidth=2)

    for name, y_pred in pred_dict.items():
        plt.plot(y_pred[:num_points], label=f"{name} Prediction", alpha=0.8)

    plt.xlabel("Time Step")
    plt.ylabel("PM2.5")
    plt.title("Figure 4. Prediction Comparison of Different Models")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, s=12, alpha=0.5, label="Residuals")
    plt.axhline(0, linestyle="--", label="Zero Error Line")
    plt.xlabel("Predicted PM2.5")
    plt.ylabel("Residual")
    plt.title(f"Figure 5. Residual Plot of {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_true, y_pred, model_name):
    errors = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, alpha=0.8, label="Prediction Error")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"Figure 6. Error Distribution of {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(result_df, metric):
    plt.figure(figsize=(8, 5))
    plt.bar(result_df["Model"], result_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Figure 7. Model Comparison Based on {metric}")
    plt.tight_layout()
    plt.show()


# =========================================================
# 11. 主程序
# =========================================================

def main():
    # -----------------------------
    # 数据读取与预处理
    # -----------------------------
    raw_data, scaled_data, scaler, feature_cols = load_and_preprocess_data(DATA_PATH)

    plot_raw_pm25(raw_data)

    # -----------------------------
    # 构造监督学习样本
    # -----------------------------
    X, y = create_sequences(
        data_array=scaled_data,
        look_back=LOOK_BACK,
        target_col=0
    )

    print("=" * 80)
    print("监督学习样本构造完成：")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # -----------------------------
    # 划分训练集和测试集
    # -----------------------------
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        train_ratio=TRAIN_RATIO
    )

    print("=" * 80)
    print("训练集与测试集：")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])
    n_features = scaled_data.shape[1]

    # -----------------------------
    # 回调函数
    # -----------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        verbose=1
    )

    # -----------------------------
    # 训练 LSTM
    # -----------------------------
    print("=" * 80)
    print("开始训练 LSTM 模型")

    lstm_model = build_lstm_model(input_shape)
    lstm_model.summary()

    lstm_history = lstm_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    plot_training_loss(lstm_history, "LSTM")

    # -----------------------------
    # 训练 GRU
    # -----------------------------
    print("=" * 80)
    print("开始训练 GRU 模型")

    gru_model = build_gru_model(input_shape)
    gru_model.summary()

    gru_history = gru_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    plot_training_loss(gru_history, "GRU")

    # -----------------------------
    # 训练 XGBoost
    # -----------------------------
    xgb_model = None

    if XGBOOST_AVAILABLE:
        print("=" * 80)
        print("开始训练 XGBoost 模型")

        X_train_xgb = flatten_sequence_data(X_train)
        X_test_xgb = flatten_sequence_data(X_test)

        xgb_model = build_xgboost_model()
        xgb_model.fit(X_train_xgb, y_train)
    else:
        print("=" * 80)
        print("未安装 xgboost，将跳过 XGBoost 模型。")
        print("如需使用 XGBoost，请运行：pip install xgboost")

    # -----------------------------
    # 预测与反归一化
    # -----------------------------
    y_test_inv = inverse_transform_target(
        y_test,
        scaler,
        n_features=n_features,
        target_col=0
    )

    predictions = {}

    lstm_pred_scaled = lstm_model.predict(X_test)
    lstm_pred_inv = inverse_transform_target(
        lstm_pred_scaled,
        scaler,
        n_features=n_features,
        target_col=0
    )
    predictions["LSTM"] = lstm_pred_inv

    gru_pred_scaled = gru_model.predict(X_test)
    gru_pred_inv = inverse_transform_target(
        gru_pred_scaled,
        scaler,
        n_features=n_features,
        target_col=0
    )
    predictions["GRU"] = gru_pred_inv

    if xgb_model is not None:
        xgb_pred_scaled = xgb_model.predict(X_test_xgb)
        xgb_pred_inv = inverse_transform_target(
            xgb_pred_scaled,
            scaler,
            n_features=n_features,
            target_col=0
        )
        predictions["XGBoost"] = xgb_pred_inv

    # -----------------------------
    # 模型评价
    # -----------------------------
    result_records = []

    for name, y_pred_inv in predictions.items():
        metrics = evaluate_regression(y_test_inv, y_pred_inv)
        metrics["Model"] = name
        result_records.append(metrics)

        print("=" * 80)
        print(f"{name} 模型评价结果：")
        for k, v in metrics.items():
            if k != "Model":
                print(f"{k}: {v:.4f}")

    result_df = pd.DataFrame(result_records)
    result_df = result_df[
        ["Model", "RMSE", "MAE", "R2", "SMAPE", "Pearson"]
    ]

    print("=" * 80)
    print("模型性能对比表：")
    print(result_df)

    result_df.to_csv(
        "model_evaluation_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------
    # 可视化
    # -----------------------------
    plot_prediction_curve(y_test_inv, predictions["LSTM"], "LSTM", num_points=300)
    plot_all_predictions(y_test_inv, predictions, num_points=300)

    plot_residuals(y_test_inv, predictions["LSTM"], "LSTM")
    plot_error_distribution(y_test_inv, predictions["LSTM"], "LSTM")

    plot_metric_comparison(result_df, "RMSE")
    plot_metric_comparison(result_df, "MAE")
    plot_metric_comparison(result_df, "R2")
    plot_metric_comparison(result_df, "SMAPE")
    plot_metric_comparison(result_df, "Pearson")

    # -----------------------------
    # 保存预测结果
    # -----------------------------
    pred_df = pd.DataFrame({
        "True_PM2.5": y_test_inv
    })

    for name, pred in predictions.items():
        pred_df[f"{name}_Prediction"] = pred

    pred_df.to_csv(
        "prediction_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("实验完成。")
    print("评价结果已保存为 model_evaluation_results.csv")
    print("预测结果已保存为 prediction_results.csv")


if __name__ == "__main__":
    main()