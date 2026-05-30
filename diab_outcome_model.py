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
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_cleaned.csv")

# split features and target
X = df.drop(columns=['readmitted_binary', 'any_readmission_binary'])
#y = df['readmitted_binary']
y = df['any_readmission_binary'] # alternative target to try, commment out one of the y's


# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# logistic regression baseline
#lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # use with unbalanced target outcome
lr = LogisticRegression(max_iter=1000, random_state=42) # use with any readmission vs No target outcome since it is balanced
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(f"\nLogistic Regression AUC: {lr_auc:.4f}")
print(classification_report(y_test, lr.predict(X_test)))

# random forest
#rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1) # use with unbalanced target outcome
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) # use with any readmission vs No target outcome since it is balanced
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f"\nRandom Forest AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf.predict(X_test)))

# XGBoost classifier
xgb = XGBClassifier(random_state = 42,
                    #scale_pos_weight = 8, only use with original unbalanced target outcome
                    n_jobs = -1,
                    eval_metric = 'auc')
xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)
print(f"\nXGBoost AUC: {xgb_auc:.4f}")
print(classification_report(y_test, xgb.predict(X_test)))

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(y_test, lr_probs, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_predictions(y_test, rf_probs, name="Random Forest", ax=ax)
RocCurveDisplay.from_predictions(y_test, xgb_probs, name="XGBoost", ax=ax)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
ax.set_title("Diabetes Hospital Readmission Risk Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_diab_uci_LogReg_RF_XGB.png", dpi=150)
#plt.show()
