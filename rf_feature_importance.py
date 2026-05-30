# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#import matplotlib as plt
from matplotlib import pyplot as plt



df = pd.read_csv("diabetes_cleaned.csv")

# split features and target
X = df.drop(columns=['readmitted_binary', 'any_readmission_binary'])
#y = df['readmitted_binary']
y = df['any_readmission_binary'] # alternative target to try, commment out one of the y's


# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# random forest
rf = RandomForestClassifier(n_estimators=200, 
                            random_state=42, 
                            #class_weight='balanced', # only use with original unbalnaced target outcome
                              n_jobs=-1)
rf.fit(X_train, y_train)

# extract feature importance from random forest
feature_importance = pd.DataFrame({'feature': X_train.columns,'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

print('Heres a look at feature importance:')
print(feature_importance.head(20))

# plot
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance.head(20).plot(
    kind='barh',
    x='feature',
    y='importance',
    ax=ax,
    legend=False
)
ax.set_title('Top 20 Feature Importances — Diabetes Readmission')
ax.set_xlabel('Importance')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)


