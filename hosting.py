import torch
import uvicorn
import joblib
import torch.serialization
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from flask import jsonify
from main import WideDeep,WD_Config
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

torch.serialization.add_safe_globals([WideDeep,WD_Config])


#device
#device=torch.device('cpu' if torch.cuda.is_available else 'cuda')

#lets load the check point
checkpoint=torch.load('model_optim.pth',map_location=torch.device('cpu'),weights_only=True)

config=WD_Config()

config.num_product=74257
config.num_users=256042
config.num_year=14
config.num_time_day=2
config.num_month=13
config.num_day_week=7
config.embedding_dim=100

model=WideDeep(config)
model.load_state_dict(checkpoint['model_dict'])


model.eval()  #turning off drop out

app=FastAPI()
templates = Jinja2Templates(directory="templates")

class UserInput(BaseModel):
    productid:str
    user_id:str
    year:int
    month:int
    day_of_week:int
    hour:int

#load the encoder
productid_encoder=joblib.load('ProductId_encoder.pkl')
userid_encoder=joblib.load('UserId_encoder.pkl')

#lets do some preprocessing  ... it returns tensors
def preprocess(input_data):
    #encode the 
    encoded_productid=productid_encoder.transform([input_data.productid])[0]
    encoded_userid=userid_encoder.transform([input_data.user_id])[0]

    preprocessed_data=[
        encoded_productid,
        encoded_productid,
        input_data.year,
        input_data.month,
        input_data.day_of_week,
        input_data.hour
    ]
    
    return torch.tensor([preprocessed_data])

#lets define the route
@app.get("/")
async def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

#for handling the inputed text
@app.get("/items/{id}",response_class=HTMLResponse)
async def read_item(request:Request,id:str):
    return templates.TemplateResponse("item.html",{"request":request,"id":id})


@app.post('/predict')
async def predict(
    productid:str = Form(),
    user_id:str= Form(),
    year: int = Form(),
    month:int = Form(),
    day_of_week:int =Form(),
    hour:int = Form()
):
    inputData = UserInput(
        productid=productid,
        userid=user_id,
        year=year,
        month=month,
        day_of_week=day_of_week,
        hour=hour
    )

    preprocessed_data=preprocess(inputData)
    #turn of the backpropagation
    with torch.no_grad():
        prediction=model(preprocessed_data)
    
    return HTMLResponse({"prediction":prediction.item()})


if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)