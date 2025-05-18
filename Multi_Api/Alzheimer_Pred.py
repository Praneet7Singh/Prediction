import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv('alzheimer.csv')
print(df)
# Encode categorical variables
df['M_F'] = df['M_F'].map({'M': 0, 'F': 1})
le = LabelEncoder()
df['Group'] = le.fit_transform(df['Group'])

# Separate features and target
X = df.drop('Group', axis=1)
y = df['Group']

# Train/test split (split before imputation to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline with imputation and classifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the pipeline and label encoder to a pickle file
model_data = {
    'pipeline': pipeline,
    'label_encoder': le
}
with open('alzheimer_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model and label encoder saved to 'alzheimer_model.pkl'")