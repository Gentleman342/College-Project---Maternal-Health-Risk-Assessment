from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE

# Load the data and initialize the classifier outside of route functions
data = pd.read_csv('mhra.csv')
X = data.iloc[:, :-1]
y = data.RiskLevel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train_smote,y_train_smote)

pickle.dump(rf_clf, open('model.pkl','wb'))