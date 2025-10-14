import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/amazon_products_sales_data_cleaned.csv')

print(f"Data size: {data.shape}")
print(data.head().to_string())

# Преобразуем строковые значения demand в уникальные метки
label_encoder = LabelEncoder()
data['is_couponed_encoded'] = label_encoder.fit_transform(data['is_couponed'])
y = label_encoder.fit_transform(data['demand'])

features = ['reputation', 'total_price', 'listed_price', 'is_sponsored', 'is_couponed_encoded', 'buy_box_availability']
X = data[features]

# Деление на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Функция оценки модели
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"\n{name}: Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1


results = {}

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
results['KNN'] = evaluate_model('KNN', knn, X_test, y_test)
'''KNN: Accuracy: 0.8633 Precision: 0.8291 Recall: 0.8633 F1-Score: 0.8369'''

# SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
results['SVM'] = evaluate_model('SVM', svm, X_test, y_test)
'''SVM: Accuracy: 0.8651 Precision: 0.8833 Recall: 0.8651 F1-Score: 0.8025'''

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results['Random Forest'] = evaluate_model('Random Forest', rf, X_test, y_test)
'''Random Forest: Accuracy: 0.8575 Precision: 0.8290 Recall: 0.8575 F1-Score: 0.8383'''

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
results['Decision Tree'] = evaluate_model('Decision Tree', dt, X_test, y_test)
'''Decision Tree: Accuracy: 0.8186 Precision: 0.8185 Recall: 0.8186 F1-Score: 0.8185'''

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
results['Logistic Regression'] = evaluate_model('Logistic Regression', lr, X_test, y_test)
'''Logistic Regression: Accuracy: 0.8643 Precision: 0.7571 Recall: 0.8643 F1-Score: 0.8023'''

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
results['Optimized RF'] = evaluate_model('Optimized Random Forest', best_rf, X_test, y_test)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
'''Optimized Random Forest: Accuracy: 0.8705 Precision: 0.8332 Recall: 0.8705 F1-Score: 0.8294
Best params: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best score: 0.8746'''

# Гистограмма оценок моделей
results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results.keys()))
width = 0.2
for i, metric in enumerate(metrics):
    values = [results[model][i] for model in results.keys()]
    plt.bar(x + i * width, values, width, label=metric)
plt.xlabel('Модели')
plt.ylabel('Значение метрики')
plt.title('Сравнение моделей классификации')
plt.xticks(x + width * 1.5, results.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('images/models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC-AUC анализ для high demand
y_binary = (y == 2).astype(int)
y_train_binary = (y_train == 2).astype(int)
y_test_binary = (y_test == 2).astype(int)
plt.figure(figsize=(10, 8))
models_binary = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}
for name, model in models_binary.items():
    model.fit(X_train, y_train_binary)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    auc_score = roc_auc_score(y_test_binary, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые для предсказания высокого спроса')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('images/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance для Random Forest
feature_importance = rf.feature_importances_
feature_names = features
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Важность признака')
plt.title('Важность признаков (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
