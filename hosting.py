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
    userid:str
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
    encoded_userid=userid_encoder.transform([input_data.userid])[0]

    preprocessed_data={
        "productid":torch.tensor([encoded_productid]),
        "userid":torch.tensor([encoded_userid]),
        "year":torch.tensor([input_data.year]),
        "month":torch.tensor([input_data.month]),
        "day_of_week":torch.tensor([input_data.day_of_week]),
        "hour":torch.tensor(input_data.hour)
    }
    
    return preprocessed_data

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
    userid:str= Form(),
    year: int = Form(),
    month:int = Form(),
    day_of_week:int =Form(),
    hour:int = Form()
):
    
    #type checking maybe
    inputData = UserInput(
        productid=productid,
        userid=userid,
        year=year,
        month=month,
        day_of_week=day_of_week,
        hour=hour
    )

    preprocessed_data=preprocess(inputData)

    #lets unpack the
    productid=preprocessed_data['productid']
    userid=preprocessed_data['userid']
    year=preprocessed_data['year']
    month=preprocessed_data['month']
    day_of_week=preprocessed_data['day_of_week']
    hour=preprocessed_data['hour']


    #turn of the backpropagation
    with torch.no_grad():
        prediction=model(
            productid,
            userid,
            year,
            month,
            day_of_week,
            hour
        )
    
    return HTMLResponse({"prediction":prediction.item()})


if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)