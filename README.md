## Swiggy Delivery Time Prediction (ETA Prediction System)
### 📌 Project Overview

This project builds a Machine Learning model to predict food delivery time (ETA) in minutes for Swiggy orders. Accurate ETA prediction improves customer satisfaction, reduces cancellations, and optimizes delivery operations.

The system uses historical and real-time features such as traffic, weather, rider details, distance, and order information to generate reliable predictions.

### 🎯 Business Objective
* Improve ETA accuracy for better customer trust
* Reduce cancellations and refund costs
* Optimize rider allocation and delivery efficiency
* Enhance overall operational performance

### 📊 Dataset Information
* Total Records: 45,502
* Features: 20 (after preprocessing)
#### Key Features:
* Rider: Age, Ratings, Vehicle Condition
* Location: City, Distance
* External: Traffic, Weather, Festival
* Order: Type, Time, Multiple Deliveries
* Target: time_taken (Delivery Time in minutes)

### Tech Stack
* Programming: Python
* Libraries:
    pandas, numpy
    scikit-learn
    xgboost
    streamlit
* Deployment: Streamlit + Hugging Face Spaces

### ML Workflow (CRISP-ML(Q))
1) Business Understanding
2) Data Preprocessing
3) Model Building
4) Evaluation
5) Hyperparameter Tuning
6) Deployment

### Data Preprocessing
* Handled missing values using SimpleImputer
* Encoded categorical features using:
     OneHotEncoder (Nominal)
     OrdinalEncoder (Traffic levels)
* Applied:
     StandardScaler
     PowerTransformer (Yeo-Johnson)
* Used Pipeline & ColumnTransformer for automation

### Models Implemented
* Linear Regression
* KNN
* SVR
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### 🏆 Final Model: XGBoost (Tuned)
#### 🔍 Performance:
* Train R²: 0.865
* Test R²: 0.817
* Test RMSE: 4.03

### 🔁 Cross Validation
* Used K-Fold (k=5)
* Achieved stable and consistent performance
📦 Model Deployment

### 🖥️ Streamlit App Features:
### User-friendly UI
* Input fields for:
      Rider details
      Location info
      Order details
* Outputs:
      ⏱️ Predicted Delivery Time (minutes)

### 🌐 Deployment (Hugging Face Spaces)
1) Create a new Space (Streamlit)
2) Upload:
   * app.py
   * model.pkl
   * requirements.txt
3) App will auto-deploy

### 📈 Key Insights
* 🚗 Distance and traffic are the most influential features
* ⏰ Peak hours significantly increase delivery time
* 🌧️ Weather impacts ETA variability
* 📦 Multiple deliveries increase delay

### 🚀 Future Improvements
* Add real-time traffic API integration
* Use deep learning models (LSTM, ANN)
* Add uncertainty prediction (confidence intervals)
* Implement real-time ETA updates
