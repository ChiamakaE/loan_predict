from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class ModelInput(BaseModel):
    Gender: str
    Married: str
    Dependents: object
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

 # Load the trained model
loan_predict = joblib.load("pipe_logit.pkl")


@app.post('/loan_predict')
def predict_loan(model_input: ModelInput):
    input_dict = model_input.dict()
    input_df = pd.DataFrame([input_dict])
    input_list = input_df.values  # Get the column values as a NumPy array


# Get the column values excluding the first column

    input_list = [[
        input_dict['Gender'],
        input_dict['Married'],
        input_dict['Dependents'],
        input_dict['Education'],
        input_dict['Self_Employed'],
        input_dict['ApplicantIncome'],
        input_dict['CoapplicantIncome'],
        input_dict['LoanAmount'],
        input_dict['Loan_Amount_Term'],
        input_dict['Credit_History'],
        input_dict['Property_Area'],
       
    ]]

   
    # Make predictions
    prediction = loan_predict.predict(input_df)

    # Prepare the response
    if (prediction[0] == 1):
        return 'Applicant will not default'
    else:
        return 'Applicant will default'

   

    
