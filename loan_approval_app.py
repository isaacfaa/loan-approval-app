import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib

url_kaggle = "https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset"
url_gbc = "https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"

col1, col2, col3 = st.columns(3)

st.markdown("<h1 style='text-align: center;'>💻🧮 Loan Approval Predictor 🧮💻</h1>", unsafe_allow_html=True)
st.write("""
Do you want to predict if your loan would be approved? You can use this app to find out! With a mmachine learning algorithm, you can predict if your loan would be approved or not by just adding in some details such as your annual income and how much you wish to loan
""")


st.sidebar.header('User Input Parameters')

def user_input_features():
    income_annum = st.sidebar.number_input('Income Per Annum:', min_value=0, step=1000)
    loan_ammount = st.sidebar.number_input('Loan Ammount', min_value=0, step=1000)
    loan_term = st.sidebar.number_input('Loan Term', min_value=2, max_value=20, step=1)
    cibil_score = st.sidebar.number_input('Cibil Score', min_value=0, step=1)
    data = {'income_annum': income_annum,
            'loan_ammount': loan_ammount,
            'loan_term': loan_term,
            'cibil_score': cibil_score}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.sidebar.header("Links:")
st.sidebar.write("[Kaggle Dataset](%s)" % url_kaggle)
st.sidebar.write("[Model Used](%s)" % url_gbc)

st.markdown("<h2 style='text-align: center;'>Parameters added for prediction</h2>", unsafe_allow_html=True)
st.write(df)

lad = pd.read_csv('updated_loan_approval_dataset.csv')


model = joblib.load('trained_loan_approval_model.pkl')


prediction = model.predict(df)
if prediction == 1:
    prediction = "Approved!"
else:
    prediction = "Rejected :("
prediction_proba = model.predict_proba(df)

st.subheader('Prediction ✨:')
st.write(prediction)

st.markdown("<h1 style='text-align: center;'>📝 Information About the Loan Approval App 📝</h1>", unsafe_allow_html=True)
st.subheader('Dataset')
st.write('The dataset used for training the model was downloaded from [Kaggle](%s)' % url_kaggle)
st.subheader("About Dataset (From Kaggle)")
st.write("The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. This dataset is commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.")

st.subheader('Dataset visualisation')
st.write(lad)

st.subheader('Model Information')
st.write("The Gradient Boosting Classifier algorithm builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case where only a single regression tree is induced.")
st.write("[Documentation of Model used](%s)" % "https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html")