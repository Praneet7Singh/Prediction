import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle 

data = pd.read_csv('Dataset of Diabetes .csv')

print(data)
data = data.drop(['ID', 'No_Pation'], axis=1)

data['Gender'] = data['Gender'].str.upper().map({'F':0, 'M':1})
data['CLASS'] = data['CLASS'].str.upper().map({'N':0, 'P':1, 'Y':2})


imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data['Gender'] = imputer.fit_transform(data['Gender'].values.reshape(-1, 1)).ravel()
data['CLASS'] = imputer.fit_transform(data['CLASS'].values.reshape(-1, 1)).ravel()

X = data.drop('CLASS', axis=1)
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler, 'columns': X.columns.tolist()}, f)