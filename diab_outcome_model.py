# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, 
                             classification_report, 
                             RocCurveDisplay, 
                             confusion_matrix)
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_cleaned.csv")

# split features and target
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# logistic regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(f"\nLogistic Regression AUC: {lr_auc:.4f}")
print(classification_report(y_test, lr.predict(X_test)))

# random forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f"\nRandom Forest AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf.predict(X_test)))

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, lr_probs, name=f"Logistic Regression (AUC={lr_auc:.3f})", ax=ax)
RocCurveDisplay.from_predictions(y_test, rf_probs, name=f"Random Forest (AUC={rf_auc:.3f})", ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
ax.set_title("Diabetes 30-Day Readmission Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_diab_uci_LogReg_RF.png", dpi=150)
#plt.show()
