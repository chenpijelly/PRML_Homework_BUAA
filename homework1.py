import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 读取数据 ====================
train_data = pd.read_excel('D:\桌面文件\自动化大三下\\1-课内课程\\5-模式识别与机器学习\Data4Regression.xlsx', sheet_name='Training Data')
test_data = pd.read_excel('D:\桌面文件\自动化大三下\\1-课内课程\\5-模式识别与机器学习\Data4Regression.xlsx', sheet_name='Test Data')

# 清理数据
train_data = train_data[train_data['x'] != 'x'].astype(float)
test_data = test_data[test_data['x_new'] != 'x_new'].astype(float)

X_train = train_data['x'].values
y_train = train_data['y_complex'].values
X_test = test_data['x_new'].values
y_test = test_data['y_new_complex'].values

# ==================== 最小二乘法 ====================
def least_squares(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    theta1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    theta0 = y_mean - theta1 * X_mean
    return theta0, theta1

# ==================== 梯度下降法 ====================
def gradient_descent(X, y, lr=0.01, epochs=5000):
    theta0, theta1 = 0, 0
    n = len(X)
    for _ in range(epochs):
        y_pred = theta0 + theta1 * X
        d_theta0 = (2/n) * np.sum(y_pred - y)
        d_theta1 = (2/n) * np.sum((y_pred - y) * X)
        theta0 -= lr * d_theta0
        theta1 -= lr * d_theta1
    return theta0, theta1

# ==================== 牛顿法 ====================
def newton_method(X, y):
    n = len(X)
    X_mean = np.mean(X)
    theta1 = np.sum((X - X_mean) * (y - np.mean(y))) / np.sum((X - X_mean) ** 2)
    theta0 = np.mean(y) - theta1 * X_mean
    return theta0, theta1

# ==================== 计算误差 ====================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 训练三种模型
theta0_ls, theta1_ls = least_squares(X_train, y_train)
theta0_gd, theta1_gd = gradient_descent(X_train, y_train, lr=0.01, epochs=5000)
theta0_nt, theta1_nt = newton_method(X_train, y_train)

# 生成预测
x_line = np.linspace(0, 10, 100)
y_ls = theta0_ls + theta1_ls * x_line
y_gd = theta0_gd + theta1_gd * x_line
y_nt = theta0_nt + theta1_nt * x_line

# 计算误差
pred_train_ls = theta0_ls + theta1_ls * X_train
pred_test_ls = theta0_ls + theta1_ls * X_test
pred_train_gd = theta0_gd + theta1_gd * X_train
pred_test_gd = theta0_gd + theta1_gd * X_test
pred_train_nt = theta0_nt + theta1_nt * X_train
pred_test_nt = theta0_nt + theta1_nt * X_test

mse_ls_train = mse(y_train, pred_train_ls)
mse_ls_test = mse(y_test, pred_test_ls)
mse_gd_train = mse(y_train, pred_train_gd)
mse_gd_test = mse(y_test, pred_test_gd)
mse_nt_train = mse(y_train, pred_train_nt)
mse_nt_test = mse(y_test, pred_test_nt)

# ==================== 图1：数据分布 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.6, s=40, label='Training Data', edgecolors='white', linewidth=0.5)
plt.scatter(X_test, y_test, c='red', alpha=0.6, s=40, label='Test Data', edgecolors='white', linewidth=0.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Data Distribution', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_data_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图2：最小二乘法 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_line, y_ls, 'g-', linewidth=2.5, label='Fitted Line')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Least Squares Method\nTrain MSE = {mse_ls_train:.4f}, Test MSE = {mse_ls_test:.4f}',
          fontsize=14, fontweight='bold', color='green')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_least_squares.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图3：梯度下降法 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_line, y_gd, 'purple', linewidth=2.5, label='Fitted Line')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Gradient Descent Method\nTrain MSE = {mse_gd_train:.4f}, Test MSE = {mse_gd_test:.4f}',
          fontsize=14, fontweight='bold', color='purple')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_gradient_descent.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图4：牛顿法 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_line, y_nt, 'orange', linewidth=2.5, label='Fitted Line')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Newton Method\nTrain MSE = {mse_nt_train:.4f}, Test MSE = {mse_nt_test:.4f}',
          fontsize=14, fontweight='bold', color='darkorange')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_newton_method.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 打印结果汇总 ====================
print("=" * 60)
print("三种线性回归方法结果对比")
print("=" * 60)
print(f"{'方法':<20} {'训练MSE':<15} {'测试MSE':<15}")
print("-" * 60)
print(f"{'Least Squares':<20} {mse_ls_train:<15.6f} {mse_ls_test:<15.6f}")
print(f"{'Gradient Descent':<20} {mse_gd_train:<15.6f} {mse_gd_test:<15.6f}")
print(f"{'Newton Method':<20} {mse_nt_train:<15.6f} {mse_nt_test:<15.6f}")
print("=" * 60)