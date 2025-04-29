# Salary Prediction Web App

This project is a machine learning-powered web application that predicts an employee's salary based on multiple input features including age, gender, designation, department, experience, and performance rating. The model is built using TensorFlow and deployed with Streamlit to create an interactive frontend.

## Features

- Predicts salary based on real employee attributes
- Includes preprocessing with Label Encoding and One-Hot Encoding
- Feature scaling using StandardScaler
- Trained using a Sequential Keras model
- User-friendly interface powered by Streamlit

## Model Inputs

The model requires the following input features:
- SEX (Gender)
- DESIGNATION (Job Role)
- AGE (in Years)
- UNIT (Department)
- RATINGS (Performance Rating)
- PAST EXP (Years of Previous Experience)

## Target Variable

- SALARY (The model predicts this numeric value)

## Project Structure

```
├── model_tr.ipynb              # Jupyter Notebook for training the model
├── prediction.py               # Streamlit application for salary prediction
├── model.h5 / model.keras      # Trained Keras model
├── scaler.pkl                  # StandardScaler object
├── label_encoder_sex.pkl       # Label encoder for SEX
├── onehot_encoder_des.pkl      # One-hot encoder for DESIGNATION
├── onehot_encoder_unit.pkl     # One-hot encoder for UNIT
├── feature_order.pkl           # Preserved order of features from training
├── salary prediction.csv       # Dataset file
```

## How to Run the App

1. Clone the repository:
   ```
   git clone https://github.com/mharoon1578/salary-prediction.git
   cd salary-prediction
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   ```
   streamlit run prediction.py
   ```

## Training the Model

To retrain the model:

1. Open the `model_tr.ipynb` notebook
2. Run all cells sequentially
3. The following artifacts will be saved:
   - Trained model file
   - Encoders and scaler
   - Feature order file

## Example Prediction

Input:
- SEX: Female
- AGE: 21
- DESIGNATION: Analyst
- UNIT: Finance
- RATINGS: 2.0
- PAST EXP: 0

Predicted Salary: 60,783.07  
Actual Salary: 57,488  
Relative Error: Approximately 5.73%

## Built With

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook

## License

This project is open-source and free to use under the terms of the MIT license.

## Author

Muhammad Haroon  
GitHub: [@mharoon1578](https://github.com/mharoon1578)
