# Kidney Stone Prediction (Bi-LSTM)

This project predicts the risk of kidney stone formation using clinical lab values such as Specific Gravity, Urine pH, Osmolality, Conductivity, Urea, and Calcium. It uses a Bidirectional LSTM (Bi-LSTM) deep learning model (~78.5% validation accuracy) and provides an interactive Streamlit web app for real-time prediction.

## Features
- Preprocessing & feature engineering with Pandas & NumPy
- Bi-LSTM model built using TensorFlow/Keras
- Exploratory analysis & visualization (Jupyter Notebooks)
- Streamlit app for user-friendly predictions

## Installation
git clone https://github.com/bandarupalli-raviteja/Raviteja.B.git
cd Raviteja.B
pip install -r requirements.txt

## Usage
Train the model:
python train_bilstm.py

Run the Streamlit app:
streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.

## Author
Raviteja B  
GitHub Profile: https://github.com/bandarupalli-raviteja
