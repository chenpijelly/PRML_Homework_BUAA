import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc


# =========================================================
# 1. 数据生成函数
# =========================================================
def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y


# =========================================================
# 2. 可视化函数
# =========================================================
def plot_3d_dataset(X, y, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter0 = ax.scatter(
        X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
        label='Class C0', s=20
    )
    scatter1 = ax.scatter(
        X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
        label='Class C1', s=20
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pca_projection(X_train, y_train, X_test, y_test):
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1],
        alpha=0.6, s=30, label='Train C0'
    )
    plt.scatter(
        X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1],
        alpha=0.6, s=30, label='Train C1'
    )
    plt.scatter(
        X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
        marker='x', s=50, label='Test C0'
    )
    plt.scatter(
        X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
        marker='x', s=50, label='Test C1'
    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA 2D Projection of Train/Test Data')
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 3. 模型定义
# =========================================================
def make_adaboost_with_tree(estimator, n_estimators=100, learning_rate=1.0, random_state=42):
    # 兼容不同版本 sklearn
    try:
        model = AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    except TypeError:
        model = AdaBoostClassifier(
            base_estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    return model


def get_models():
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ),
        'AdaBoost + Tree': make_adaboost_with_tree(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        'SVM Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        'SVM Poly': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='poly', degree=3, C=1.0, gamma='scale', probability=True, random_state=42))
        ]),
        'SVM RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ]),
    }
    return models


# =========================================================
# 4. 初始模型性能对比
# =========================================================
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc
        })

    df_results = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    return df_results


# =========================================================
# 5. ROC 曲线
# =========================================================
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Initial Models')
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. 调参函数
# =========================================================
def tune_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 8, 10],
        'min_samples_split': [2, 5, 10, 20]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_adaboost(X_train, y_train):
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

    try:
        ada = AdaBoostClassifier(estimator=base_tree, random_state=42)
    except TypeError:
        ada = AdaBoostClassifier(base_estimator=base_tree, random_state=42)

    param_grid = {
        'n_estimators': [30, 50, 100, 150, 200],
        'learning_rate': [0.1, 0.5, 1.0, 1.5]
    }

    grid = GridSearchCV(
        ada,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_svm_linear(X_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', probability=True, random_state=42))
    ])

    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_svm_poly(X_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='poly', probability=True, random_state=42))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__degree': [2, 3, 4],
        'svc__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_svm_rbf(X_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42))
    ])

    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


# =========================================================
# 7. 调参热力图
# =========================================================
def plot_heatmap_from_pivot(pivot, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean CV Accuracy')
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f'{pivot.iloc[i, j]:.3f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()


def plot_gridsearch_heatmap_decision_tree(grid):
    results = pd.DataFrame(grid.cv_results_)
    pivot = results.pivot_table(
        values='mean_test_score',
        index='param_min_samples_split',
        columns='param_max_depth',
        aggfunc='mean'
    )
    plot_heatmap_from_pivot(
        pivot,
        'Decision Tree Grid Search Heatmap',
        'max_depth',
        'min_samples_split'
    )


def plot_gridsearch_heatmap_adaboost(grid):
    results = pd.DataFrame(grid.cv_results_)
    pivot = results.pivot_table(
        values='mean_test_score',
        index='param_learning_rate',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    plot_heatmap_from_pivot(
        pivot,
        'AdaBoost Grid Search Heatmap',
        'n_estimators',
        'learning_rate'
    )


def plot_gridsearch_heatmap_svm_linear(grid):
    results = pd.DataFrame(grid.cv_results_)
    pivot = pd.DataFrame(results[['param_svc__C', 'mean_test_score']])
    pivot['dummy'] = 'score'
    pivot = pivot.pivot(index='dummy', columns='param_svc__C', values='mean_test_score')

    plot_heatmap_from_pivot(
        pivot,
        'SVM Linear Grid Search Heatmap',
        'C',
        ''
    )


def plot_gridsearch_heatmap_svm_poly(grid):
    results = pd.DataFrame(grid.cv_results_)
    pivot = results.pivot_table(
        values='mean_test_score',
        index='param_svc__degree',
        columns='param_svc__C',
        aggfunc='mean'
    )
    plot_heatmap_from_pivot(
        pivot,
        'SVM Poly Grid Search Heatmap (averaged over gamma)',
        'C',
        'degree'
    )


def plot_gridsearch_heatmap_svm_rbf(grid):
    results = pd.DataFrame(grid.cv_results_)
    results['param_svc__gamma'] = results['param_svc__gamma'].astype(str)
    results['param_svc__C'] = results['param_svc__C'].astype(str)

    pivot = results.pivot_table(
        values='mean_test_score',
        index='param_svc__gamma',
        columns='param_svc__C',
        aggfunc='mean'
    )
    plot_heatmap_from_pivot(
        pivot,
        'SVM RBF Grid Search Heatmap',
        'C',
        'gamma'
    )


# =========================================================
# 8. 主程序
# =========================================================
def main():
    np.random.seed(42)

    # 训练集：1000个样本（每类500）
    X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)

    # 测试集：500个样本（每类250）
    X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

    # -------------------------
    # 训练集与测试集绘制
    # -------------------------
    plot_3d_dataset(X_train, y_train, 'Training Data (3D)')
    plot_3d_dataset(X_test, y_test, 'Test Data (3D)')

    # -------------------------
    # PCA 二维降维可视化
    # -------------------------
    plot_pca_projection(X_train, y_train, X_test, y_test)

    # -------------------------
    # 初始模型性能对比
    # -------------------------
    models = get_models()
    df_results = evaluate_models(models, X_train, y_train, X_test, y_test)

    print("=" * 80)
    print("Initial Model Performance Comparison")
    print(df_results.to_string(index=False))

    # -------------------------
    # ROC 曲线图
    # -------------------------
    plot_roc_curves(models, X_test, y_test)

    # -------------------------
    # 调参热力图
    # -------------------------
    print("=" * 80)
    print("Start GridSearchCV...")

    grid_dt = tune_decision_tree(X_train, y_train)
    print("Decision Tree best params:", grid_dt.best_params_)
    print("Decision Tree best CV score:", grid_dt.best_score_)
    plot_gridsearch_heatmap_decision_tree(grid_dt)

    grid_ada = tune_adaboost(X_train, y_train)
    print("AdaBoost best params:", grid_ada.best_params_)
    print("AdaBoost best CV score:", grid_ada.best_score_)
    plot_gridsearch_heatmap_adaboost(grid_ada)

    grid_linear = tune_svm_linear(X_train, y_train)
    print("SVM Linear best params:", grid_linear.best_params_)
    print("SVM Linear best CV score:", grid_linear.best_score_)
    plot_gridsearch_heatmap_svm_linear(grid_linear)

    grid_poly = tune_svm_poly(X_train, y_train)
    print("SVM Poly best params:", grid_poly.best_params_)
    print("SVM Poly best CV score:", grid_poly.best_score_)
    plot_gridsearch_heatmap_svm_poly(grid_poly)

    grid_rbf = tune_svm_rbf(X_train, y_train)
    print("SVM RBF best params:", grid_rbf.best_params_)
    print("SVM RBF best CV score:", grid_rbf.best_score_)
    plot_gridsearch_heatmap_svm_rbf(grid_rbf)


if __name__ == '__main__':
    main()