import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

# ==================== 读取数据 ====================
train_data = pd.read_excel('D:\桌面文件\自动化大三下\\1-课内课程\\5-模式识别与机器学习\Data4Regression.xlsx', sheet_name='Training Data')
test_data = pd.read_excel('D:\桌面文件\自动化大三下\\1-课内课程\\5-模式识别与机器学习\Data4Regression.xlsx', sheet_name='Test Data')

# 清理数据
train_data = train_data[train_data['x'] != 'x'].astype(float)
test_data = test_data[test_data['x_new'] != 'x_new'].astype(float)

X_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y_complex'].values
X_test = test_data['x_new'].values.reshape(-1, 1)
y_test = test_data['y_new_complex'].values

# ==================== 计算误差 ====================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ==================== 模型1：线性回归（基准） ====================
def linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    theta1 = np.sum((X.flatten() - X_mean) * (y - y_mean)) / np.sum((X.flatten() - X_mean) ** 2)
    theta0 = y_mean - theta1 * X_mean
    return theta0, theta1

theta0_lin, theta1_lin = linear_regression(X_train, y_train)
pred_lin_train = theta0_lin + theta1_lin * X_train.flatten()
pred_lin_test = theta0_lin + theta1_lin * X_test.flatten()
mse_lin_train = mse(y_train, pred_lin_train)
mse_lin_test = mse(y_test, pred_lin_test)
r2_lin = r2_score(y_test, pred_lin_test)

# ==================== 模型2：多项式回归 ====================
# 使用10阶多项式 + Ridge正则化
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=10, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=0.1))
])
poly_model.fit(X_train, y_train)
pred_poly_train = poly_model.predict(X_train)
pred_poly_test = poly_model.predict(X_test)
mse_poly_train = mse(y_train, pred_poly_train)
mse_poly_test = mse(y_test, pred_poly_test)
r2_poly = r2_score(y_test, pred_poly_test)

# ==================== 模型3：RBF核回归 ====================
# 使用RBF核函数将数据映射到高维空间
rbf_feature = RBFSampler(gamma=0.5, n_components=200, random_state=42)
X_train_rbf = rbf_feature.fit_transform(X_train)
X_test_rbf = rbf_feature.transform(X_test)

rbf_model = Ridge(alpha=0.01)
rbf_model.fit(X_train_rbf, y_train)
pred_rbf_train = rbf_model.predict(X_train_rbf)
pred_rbf_test = rbf_model.predict(X_test_rbf)
mse_rbf_train = mse(y_train, pred_rbf_train)
mse_rbf_test = mse(y_test, pred_rbf_test)
r2_rbf = r2_score(y_test, pred_rbf_test)

# ==================== 生成平滑曲线用于绘图 ====================
x_smooth = np.linspace(0, 10, 500).reshape(-1, 1)
y_lin_smooth = theta0_lin + theta1_lin * x_smooth.flatten()
y_poly_smooth = poly_model.predict(x_smooth)
y_rbf_smooth = rbf_model.predict(rbf_feature.transform(x_smooth))

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

# ==================== 图2：线性回归（基准） ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_smooth, y_lin_smooth, 'gray', linewidth=2.5, label='Linear Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Linear Regression (Baseline)\nTrain MSE={mse_lin_train:.4f}, Test MSE={mse_lin_test:.4f}, R²={r2_lin:.4f}',
          fontsize=14, fontweight='bold', color='gray')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_linear_baseline.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图3：多项式回归 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_smooth, y_poly_smooth, 'green', linewidth=2.5, label='Polynomial Fit (degree=10)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Polynomial Regression\nTrain MSE={mse_poly_train:.4f}, Test MSE={mse_poly_test:.4f}, R²={r2_poly:.4f}',
          fontsize=14, fontweight='bold', color='green')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_polynomial.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图4：RBF核回归 ====================
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='blue', alpha=0.4, s=35, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.4, s=35, label='Test Data')
plt.plot(x_smooth, y_rbf_smooth, 'purple', linewidth=2.5, label='RBF Kernel Fit')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'RBF Kernel Regression\nTrain MSE={mse_rbf_train:.4f}, Test MSE={mse_rbf_test:.4f}, R²={r2_rbf:.4f}',
          fontsize=14, fontweight='bold', color='purple')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_rbf_kernel.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 图5：三种方法对比 ====================
plt.figure(figsize=(12, 7))
plt.scatter(X_train, y_train, c='blue', alpha=0.3, s=30, label='Training Data')
plt.scatter(X_test, y_test, c='red', alpha=0.3, s=30, label='Test Data')
plt.plot(x_smooth, y_lin_smooth, 'gray', linewidth=2, linestyle='--', label=f'Linear (R²={r2_lin:.3f})')
plt.plot(x_smooth, y_poly_smooth, 'green', linewidth=2, label=f'Polynomial (R²={r2_poly:.3f})')
plt.plot(x_smooth, y_rbf_smooth, 'purple', linewidth=2, label=f'RBF Kernel (R²={r2_rbf:.3f})')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Model Comparison: Linear vs Polynomial vs RBF Kernel', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()