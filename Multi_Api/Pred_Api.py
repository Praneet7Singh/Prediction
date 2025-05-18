import pickle 
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import shap
from typing import Optional
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate, false_positive_rate, false_negative_rate
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score
import os

with open('heart_disease_model.pkl', 'rb') as f:
    data1 = pickle.load(f)

hd_model = data1['model']
hd_scaler = data1['scaler']

with open('diabetes_model.pkl', 'rb') as f:
    data2 = pickle.load(f)
    
db_model = data2['model']
db_scaler = data2['scaler']
db_columns = data2['columns']

with open('alzheimer_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
PIPELINE = model_data['pipeline']
LABEL_ENCODER = model_data['label_encoder']

df = pd.read_csv('alzheimer.csv')
df['M_F'] = df['M_F'].map({'M': 0, 'F': 1})
X = df.drop('Group', axis=1)
X_imputed = PIPELINE.named_steps['imputer'].transform(X)
EXPLAINER = shap.TreeExplainer(PIPELINE.named_steps['classifier'], X_imputed)  
    
with open('common_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)
    vectorizer = saved_model['vectorizer']
    clf = saved_model['classifier']    
    
app=FastAPI()

origins = [
    "http://localhost:4200",  
    "http://127.0.0.1:4200"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

# Initialize explainers after loading models
hd_explainer = shap.TreeExplainer(hd_model)
db_explainer = shap.TreeExplainer(db_model)

# Define input data model
class H_PatientData(BaseModel):
    age: int
    gender: int  # 0 = female, 1 = male
    chest_pain: int  # 0-3
    resting_bp: int
    cholesterol: float
    fasting_blood_sugar: int  # 0 or 1
    resting_ecg: int  # 0-2
    max_heart_rate: int
    exercise_angina: int  # 0 or 1
    oldpeak: float
    slope: int  # 0-3
    major_vessels: int  # 0-3


class D_PatientData(BaseModel):
    Gender: int
    AGE: float
    Urea: float
    Cr: float
    HbA1c: float
    Chol: float
    TG: float
    HDL: float
    LDL: float
    VLDL: float
    BMI: float

class AlzheimerInput(BaseModel):
    M_F: Optional[float] = Field(None, alias='M_F') 
    Age: Optional[float]
    EDUC: Optional[float]
    SES: Optional[float]
    MMSE: Optional[float]
    CDR: Optional[float]
    eTIV: Optional[float]
    nWBV: Optional[float]
    ASF: Optional[float]

class SymptomsInput(BaseModel):
    symptoms: str    
#prediction  
@app.post("/cardio_predict")
def predict(data1: H_PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        'age': data1.age,
        'gender': data1.gender,
        'chestpain': data1.chest_pain,
        'restingBP': data1.resting_bp,
        'serumcholestrol': data1.cholesterol,
        'fastingbloodsugar': data1.fasting_blood_sugar,
        'restingrelectro': data1.resting_ecg,
        'maxheartrate': data1.max_heart_rate,
        'exerciseangia': data1.exercise_angina,
        'oldpeak': data1.oldpeak,
        'slope': data1.slope,
        'noofmajorvessels': data1.major_vessels
    }])

    # Scale input
    input_scaled = hd_scaler.transform(input_df)

    # Predict probability and class
    proba = hd_model.predict_proba(input_scaled)[0, 1]
    prediction = int(proba >= 0.5)

    # Risk level
    if proba < 0.25:
        risk_level = "Low"
    elif proba < 0.5:
        risk_level = "Moderate"
    elif proba < 0.75:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "probability": float(proba),
        "prediction": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "risk_level": risk_level
    }

  
@app.post("/diabetes_predict")
def predict_diabetes(patient: D_PatientData):
    # Prepare input data
    input_df = pd.DataFrame([[patient.Gender, patient.AGE, patient.Urea, patient.Cr, patient.HbA1c,
                              patient.Chol, patient.TG, patient.HDL, patient.LDL, patient.VLDL, patient.BMI]],
                            columns=db_columns)
    input_scaled = db_scaler.transform(input_df)
    proba = db_model.predict_proba(input_scaled)[0]
    pred = db_model.predict(input_scaled)[0]
    class_map = {0: 'NonDiabetic', 1: 'Prediabetic', 2: 'Diabetic'}
    # Convert np.float64 to native float
    proba_dict = {class_map[i]: float(prob) for i, prob in enumerate(proba)}
    return {
        'predicted_class': class_map[pred],
        'probabilities': proba_dict
    }
    
@app.post("/alzheimer_predict")
def predict_alzheimer(input_data: AlzheimerInput):
     # Convert input data to dictionary using aliases
    data_dict = input_data.dict(by_alias=True)
    # Create a DataFrame from the input data
    df = pd.DataFrame([data_dict])
    # Make prediction using the pipeline
    prediction = PIPELINE.predict(df)[0]
    # Get probabilities for each class
    probabilities = PIPELINE.predict_proba(df)[0]
    # Get the class labels from the classifier
    classes = PIPELINE.named_steps['classifier'].classes_
    # Map probabilities to their respective classes
    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
    # Inverse transform the predicted class and probabilities to original labels
    original_label = LABEL_ENCODER.inverse_transform([prediction])[0]
    original_prob_dict = {LABEL_ENCODER.inverse_transform([cls])[0]: prob for cls, prob in prob_dict.items()}
    # Return the prediction and probabilities
    return {
        "prediction": original_label,
        "probabilities": original_prob_dict
    }  
  
@app.post("/common_predict")
def explain_prediction_endpoint(input: SymptomsInput):
    symptoms = input.symptoms
    symptoms_tfidf = vectorizer.transform([symptoms])
    
    prediction = clf.predict(symptoms_tfidf)[0]
    predicted_class_index = list(clf.classes_).index(prediction)
    
    probabilities = clf.predict_proba(symptoms_tfidf)[0]
    predicted_probability = probabilities[predicted_class_index]
    
    # Get indices of words with non-zero TF-IDF scores
    feature_indices = symptoms_tfidf.nonzero()[1]
    
    # Get the words and their coefficients for the predicted class
    words = [vectorizer.get_feature_names_out()[i] for i in feature_indices]
    coefs = clf.coef_[predicted_class_index, feature_indices]
    
    # Pair words with their coefficients and sort by coefficient (descending)
    word_coefs = list(zip(words, coefs))
    word_coefs.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top 5 contributing words
    top_words = [word for word, coef in word_coefs[:5]]
    return {"disease": prediction, "top_words": top_words,"probability":round(float(predicted_probability), 4)}
    
#XAI    
def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

@app.post('/cardio_explain')
def explain_heart(data1: H_PatientData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        'age': data1.age,
        'gender': data1.gender,
        'chestpain': data1.chest_pain,
        'restingBP': data1.resting_bp,
        'serumcholestrol': data1.cholesterol,
        'fastingbloodsugar': data1.fasting_blood_sugar,
        'restingrelectro': data1.resting_ecg,
        'maxheartrate': data1.max_heart_rate,
        'exerciseangia': data1.exercise_angina,
        'oldpeak': data1.oldpeak,
        'slope': data1.slope,
        'noofmajorvessels': data1.major_vessels
    }])

    # Scale input
    input_scaled = hd_scaler.transform(input_df)

    # Predict probability and class
    proba = hd_model.predict_proba(input_scaled)[0, 1]
    prediction = int(proba >= 0.5)

    # Risk level
    if proba < 0.25:
        risk_level = "Low"
    elif proba < 0.5:
        risk_level = "Moderate"
    elif proba < 0.75:
        risk_level = "High"
    else:
        risk_level = "Very High"

    # Compute SHAP values
    shap_values = hd_explainer(input_scaled)

    # Generate waterfall plot
    #plt.figure(figsize=(30, 40))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    fig=plt.gcf()
    fig.set_size_inches(18,17)
    plt.title("SHAP Waterfall Plot for Heart Disease Prediction")
    waterfall_plot_base64 = plot_to_base64()
    plt.close()

    # Create SHAP contributions
    feature_names = input_df.columns.tolist()
    shap_values_pos = shap_values.values[0, :, 1]
    shap_contributions = [
        {"feature": feature, "shap_value": float(shap_val)}
        for feature, shap_val in zip(feature_names, shap_values_pos)
    ]

    return {
        "probability": float(proba),
        "prediction": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "risk_level": risk_level,
        "shap_contributions": shap_contributions,
        "waterfall_plot": waterfall_plot_base64
    }

@app.post('/diabetes_explain')
def explain_diabetes(patient: D_PatientData):
    # Step 1: Convert input patient data to a DataFrame with correct columns
    input_df = pd.DataFrame([{
        'Gender': patient.Gender,
        'AGE': patient.AGE,
        'Urea': patient.Urea,
        'Cr': patient.Cr,
        'HbA1c': patient.HbA1c,
        'Chol': patient.Chol,
        'TG': patient.TG,
        'HDL': patient.HDL,
        'LDL': patient.LDL,
        'VLDL': patient.VLDL,
        'BMI': patient.BMI
    }], columns=db_columns)

    # Step 2: Scale the input data using the pre-loaded scaler
    input_scaled = db_scaler.transform(input_df)

    # Step 3: Predict probabilities and class using the diabetes model
    proba = db_model.predict_proba(input_scaled)[0]
    pred = int(db_model.predict(input_scaled)[0])  # Predicted class index
    class_map = {0: 'NonDiabetic', 1: 'Prediabetic', 2: 'Diabetic'}
    predicted_class = class_map[pred]

    # Step 4: Compute SHAP values for the scaled input
    shap_values = db_explainer(input_scaled)

    # Step 5: Extract SHAP values and base value for the predicted class
    if isinstance(shap_values, list):
        # Older SHAP versions: list of arrays, one per class
        shap_values_pred_class = shap_values[pred][0, :]  # Shape: (n_features,)
        base_value = db_explainer.expected_value[pred]
    elif hasattr(shap_values, 'values'):
        # Newer SHAP versions: Explanation object
        shap_values_pred_class = shap_values.values[0, :, pred]  # Shape: (n_features,)
        base_value = shap_values.base_values[0, pred]
    else:
        raise ValueError("Unsupported SHAP values type from db_explainer")

    # Step 6: Generate waterfall plot for the predicted class
    explanation = shap.Explanation(
        values=shap_values_pred_class,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_size_inches(20, 18)
    plt.title(f"SHAP Waterfall Plot for Diabetes Prediction: {predicted_class}")
    waterfall_plot_base64 = plot_to_base64()
    plt.close()

    # Step 7: Create SHAP contributions for the predicted class
    feature_names = input_df.columns.tolist()
    shap_contributions = [
        {"feature": feature, "shap_value": float(shap_val)}
        for feature, shap_val in zip(feature_names, shap_values_pred_class)
    ]

    # Step 8: Convert probabilities to a dictionary
    proba_dict = {class_map[i]: float(prob) for i, prob in enumerate(proba)}

    # Step 9: Return the results
    return {
        "predicted_class": predicted_class,
        "probabilities": proba_dict,
        "shap_contributions": shap_contributions,
        "waterfall_plot": waterfall_plot_base64
    }

@app.post('/alzheimer_explain')
def explain_alzheimer(data1: AlzheimerInput):
       # Convert input to DataFrame
    input_df = pd.DataFrame([{
        'M_F': data1.M_F,
        'Age': data1.Age,
        'EDUC': data1.EDUC,
        'SES': data1.SES,
        'MMSE': data1.MMSE,
        'CDR': data1.CDR,
        'eTIV': data1.eTIV,
        'nWBV': data1.nWBV,
        'ASF': data1.ASF
    }])

    # Impute input data
    input_imputed = PIPELINE.named_steps['imputer'].transform(input_df)

    # Predict class and probabilities
    prediction = PIPELINE.named_steps['classifier'].predict(input_imputed)[0]
    probabilities = PIPELINE.named_steps['classifier'].predict_proba(input_imputed)[0]
    proba = probabilities[prediction]

    # Decode the predicted class
    original_label = LABEL_ENCODER.inverse_transform([prediction])[0]

    # Assign risk level
    if original_label == "Nondemented":
        risk_level = "Low"
    elif original_label == "Converted":
        risk_level = "Moderate"
    elif original_label == "Demented":
        risk_level = "High"
    else:
        risk_level = "Unknown"

    # Compute SHAP values
    shap_values = EXPLAINER(input_imputed)

    # Generate waterfall plot for the predicted class
    plt.figure()
    shap.plots.waterfall(shap_values[0, :, prediction], show=False)
    fig = plt.gcf()
    fig.set_size_inches(18, 17)
    plt.title("SHAP Waterfall Plot for Alzheimer’s Prediction")
    waterfall_plot_base64 = plot_to_base64()
    plt.close()

    # Create SHAP contributions for the predicted class
    feature_names = input_df.columns.tolist()
    shap_values_pos = shap_values[0, :, prediction].values  # 1D array of shape (n_features,)
    shap_contributions = [
        {"feature": feature, "shap_value": float(shap_val)}
        for feature, shap_val in zip(feature_names, shap_values_pos)
    ]

    return {
        "probability": float(proba),
        "prediction": original_label,
        "risk_level": risk_level,
        "shap_contributions": shap_contributions,
        "waterfall_plot": waterfall_plot_base64
    }
    
@app.post("/common_explain")
def common_explain(input: SymptomsInput):
    symptoms = input.symptoms

    # Transform input text to TF-IDF features
    symptoms_tfidf = vectorizer.transform([symptoms])
    
    # Get prediction and probability
    prediction = clf.predict(symptoms_tfidf)[0]
    predicted_class_index = list(clf.classes_).index(prediction)
    probabilities = clf.predict_proba(symptoms_tfidf)[0]
    predicted_probability = probabilities[predicted_class_index]
    
    # Use a zero background for SHAP explainer
    import numpy as np
    background = np.zeros((1, symptoms_tfidf.shape[1]))

    # Initialize SHAP explainer
    explainer = shap.LinearExplainer(
        clf,
        background,
        feature_names=vectorizer.get_feature_names_out()
    )
    
    # Compute SHAP values
    shap_values = explainer(symptoms_tfidf)

    # Handle multi-class and binary cases
    if len(shap_values.values.shape) == 3:
        class_shap_values = shap_values.values[0, :, predicted_class_index]
    else:
        class_shap_values = shap_values.values[0]

    # Get non-zero features in the input
    feature_indices = symptoms_tfidf.nonzero()[1]
    words = [vectorizer.get_feature_names_out()[i] for i in feature_indices]
    shap_scores = class_shap_values[feature_indices]
    
    # Sort by absolute SHAP value and get top 5
    word_shap = sorted(zip(words, shap_scores), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_features = [
        {"feature": word, "shap_value": round(float(shap_value), 6)}
        for word, shap_value in word_shap
    ]
    
    return {
        "disease": prediction,
        "top_features": top_features,
        "probability": round(float(predicted_probability), 4)
    }