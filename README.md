# 🛵 Swiggy Delivery Time Prediction (ETA Prediction System)

Predict food delivery time (in minutes) for Swiggy orders using an XGBoost regression model, served through an interactive Streamlit web app.

## 📌 Project Overview

This project builds a machine learning model to predict the delivery time (ETA) of Swiggy orders. Accurate ETA prediction improves customer satisfaction, reduces cancellations and helps optimize delivery operations.

The model uses features such as traffic, weather, rider details, distance and order information to generate its predictions.

## 🎯 Business Objective

- Improve ETA accuracy for better customer trust
- Reduce cancellations and refund costs
- Optimize rider allocation and delivery efficiency
- Enhance overall operational performance

## 📊 Dataset Information

- **Total records:** 45,502 orders (Feb–Apr 2022)
- **Raw columns:** 26
- **Model input features:** 21 (including 2 engineered features)
- **Target:** `time_taken` (delivery time in minutes)

### Key Features

| Group | Features |
|---|---|
| Rider | Age, ratings, vehicle condition, vehicle type |
| Location | City type, distance |
| External | Traffic, weather, festival |
| Order | Order type, order time, pickup time, multiple deliveries, day of week, weekend flag |

### Engineered Features

- `is_rush`: 1 when traffic is `jam` and the order is placed on a weekend
- `is_peak_hour`: 1 when the order hour is between 18 and 22

## 🧰 Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, xgboost, streamlit
- **Deployment:** Streamlit + Hugging Face Spaces

## 🔄 ML Workflow (CRISP-ML(Q))

1. Business understanding
2. Data preprocessing and feature engineering
3. Model building
4. Evaluation
5. Hyperparameter tuning
6. Deployment

## 🧹 Data Preprocessing

- Handled missing values with `SimpleImputer` (mode for categorical features, mean for numerical features)
- Encoded categorical features:
  - `OneHotEncoder` for nominal features
  - `OrdinalEncoder` for traffic level (low < medium < high < jam) and city type (semi-urban < urban < metropolitian)
- Applied `StandardScaler` and `PowerTransformer` (Yeo-Johnson) to numerical features
- Combined everything with `Pipeline` and `ColumnTransformer`, so the exact same preprocessing runs at training and prediction time

## 🤖 Models Compared

| Model | Test R² | Test RMSE (min) |
|---|---|---|
| Linear Regression | 0.566 | 6.21 |
| KNN | 0.608 | 5.90 |
| Decision Tree | 0.626 | 5.77 |
| SVR | 0.705 | 5.12 |
| Random Forest | 0.808 | 4.13 |
| **XGBoost** | **0.822** | **3.98** |

## 🏆 Final Model: XGBoost (Tuned)

Hyperparameters were tuned with `GridSearchCV` (27 combinations, 5-fold CV, scored on RMSE). Best parameters: `learning_rate=0.05`, `max_depth=8`, `n_estimators=200`.

| Metric | Train | Test |
|---|---|---|
| R² | 0.864 | 0.822 |
| RMSE (min) | 3.46 | 3.98 |

### 🔁 Cross Validation

- K-Fold (k=5) on the training set
- Mean R² = **0.814**, consistent with the test score, so there is no major overfitting

## 🖥️ Streamlit App

A simple UI where the user enters:

- **Rider details:** age, ratings, vehicle condition and type
- **Location and conditions:** city type, weather, traffic, distance
- **Order details:** order type, multiple deliveries, festival, date, pickup time, order hour

Output: ⏱️ **predicted delivery time in minutes**, with a typical error range of about ±4 minutes.

Day of week, weekend flag, time-of-day bucket, `is_rush` and `is_peak_hour` are derived automatically from the inputs, so users can't enter contradictory combinations.


### 🌐 Deploy on Hugging Face Spaces

1. Create a new Space and choose the **Streamlit** SDK
2. Upload:
   - `app.py`
   - `swiggy_delivery_time_model.pkl`
   - `swiggy_demographic.csv` 
   - `requirements.txt`
3. The app builds and deploys automatically

## 📁 Project Structure

```
├── app.py                           # Streamlit app
├── swiggy_delivery_time_model.pkl   # Trained pipeline (XGBoost)
├── swiggy_demographic.csv           # Dataset                     
└── requirements.txt

## 📈 Key Insights

- 🚗 Distance and traffic are expected to be the most influential features
- ⏰ Peak hours tend to increase delivery time
- 🌧️ Weather adds variability to the ETA
- 📦 Multiple deliveries increase delay

## 🚀 Future Improvements

- Integrate real-time traffic and weather APIs
- Try deep learning models (ANN, LSTM)
- Add uncertainty estimates (prediction intervals)
- Support real-time ETA updates
