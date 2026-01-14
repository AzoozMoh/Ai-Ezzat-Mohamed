import pandas as pd
import numpy as np


# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
# We load the dataset into a pandas DataFrame.
df = pd.read_csv("heart_disease_cleveland.csv")  # replace with the actual path or URL if needed


df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


df['ca'].replace('?', np.nan, inplace=True)
df['thal'].replace('?', np.nan, inplace=True)

df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

df = df.dropna(axis=0).reset_index(drop=True)

# At this point, df contains only complete cases.

X = df.drop('target', axis=1)
y = df['target']

categorical_features = ['cp', 'restecg', 'slope', 'thal']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Note: Binary categorical features (sex, fbs, exang) remain as is (0/1).

from sklearn.model_selection import train_test_split
# Stratify by y to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred_log = logistic_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

acc_tree = accuracy_score(y_test, y_pred_tree)
prec_tree = precision_score(y_test, y_pred_tree)
rec_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

# Print out the metrics
print("Logistic Regression Performance:")
print(f"Accuracy: {acc_log:.3f}")
print(f"Precision: {prec_log:.3f}")
print(f"Recall: {rec_log:.3f}")
print(f"F1-score: {f1_log:.3f}")

print("\nDecision Tree Performance:")
print(f"Accuracy: {acc_tree:.3f}")
print(f"Precision: {prec_tree:.3f}")
print(f"Recall: {rec_tree:.3f}")
print(f"F1-score: {f1_tree:.3f}")

# Display confusion matrices
cm_log = confusion_matrix(y_test, y_pred_log)
cm_tree = confusion_matrix(y_test, y_pred_tree)
print("\nConfusion Matrix (Logistic Regression):")
print(cm_log)   # format: [[TN, FP], [FN, TP]]
print("\nConfusion Matrix (Decision Tree):")
print(cm_tree)

feature_names = X_train.columns
log_coef = logistic_model.coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': log_coef})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print("\nTop 5 features by absolute coefficient (Logistic Regression):")
print(coef_df.head(5))

importances = tree_model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print("\nTop 5 features by importance (Decision Tree):")
print(imp_df.head(5))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Disease', 'Pred: Disease'],
            yticklabels=['Actual: No Disease', 'Actual: Disease'])
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("confusion_matrix_logistic.png")
plt.show()

plt.figure(figsize=(5,4))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Disease', 'Pred: Disease'],
            yticklabels=['Actual: No Disease', 'Actual: Disease'])
plt.title("Decision Tree - Confusion Matrix")
plt.savefig("confusion_matrix_tree.png")
plt.show()

from sklearn.tree import export_graphviz
# Export tree to DOT format for visualization (optional, requires Graphviz for rendering)
export_graphviz(tree_model, out_file="tree.dot", feature_names=feature_names,
                class_names=["NoDisease","Disease"], filled=True, rounded=True, special_characters=True)

from sklearn import tree
plt.figure(figsize=(20,10))
tree.plot_tree(tree_model, feature_names=feature_names, class_names=["No Disease","Disease"],
               filled=True, rounded=True, max_depth=3)  # max_depth=3 for a partial view
plt.title("Decision Tree (partial view, max_depth=3)")
plt.savefig("decision_tree_partial.png")
plt.show()