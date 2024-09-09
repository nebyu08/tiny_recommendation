from fastapi import FastAPI
import torch
import uvicorn
import joblib
from pydantic import BaseModel


#lets load the model
model=torch.load('')
model.eval()  #turning off drop out

app=FastAPI()

class UserInput(BaseModel):
    productid:float
    userid:float
    year:int
    month:int
    day_of_week:int
    hour:int

#load the encoder
productid_encoder=joblib.load('ProductId.pkl')
userid_encoder=joblib.load('UserId.pkl')

#lets do some preprocessing  ... it returns tensors
def preprocess(input_data):
    #encode the 
    encoded_productid=productid_encoder.transform([input_data.productid])[0]
    encoded_userid=userid_encoder.transform([input_data.userid])[0]

    preprocessed_data=[
        encoded_productid,
        encoded_productid,
        input_data.year,
        input_data.month,
        input_data.data_of_week,
        input_data.hour
    ]
    
    return torch.tensor([preprocessed_data])



@app.post('/predict')
async def predict(inputData:UserInput):
    preprocessed_data=preprocess(inputData)
    #turn of the backpropagation
    with torch.no_grad():
        prediction=model(preprocessed_data)
    
    return {"prediction":prediction}


if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)