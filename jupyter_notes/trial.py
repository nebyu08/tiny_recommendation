import torch
import torch.nn as nn
from dataclasses import dataclass

#setting up the simple configuration
@dataclass
class WD_Config:
    num_product:int=100
    num_users:int=100
    num_day_week:int=7
    num_month:int=12
    num_time_day:int=24
    num_feature:int=6  #there is no need for this
    embedding_dim:int=40
    num_year:int=2

#main model
class WideDeep(nn.Module):
    """the implementation of the wide and deep neural network based on the paper:
            link:   https://arxiv.org/pdf/1606.07792v1
    """
    def __init__(self,
                 config): 
        
        super().__init__()

        #wide
        self.wide=nn.Linear(2,1)    #2 features are product id and customer id

        #lets setup the embedding
        self.product_embed=nn.Embedding(config.num_product,config.embedding_dim)
        self.user_embed=nn.Embedding(config.num_users,config.embedding_dim)
        self.day_week_embed=nn.Embedding(config.num_day_week,config.embedding_dim)
        self.month_embed=nn.Embedding(config.num_month,config.embedding_dim)
        self.time_day_embed=nn.Embedding(config.num_time_day,config.embedding_dim)
        self.year_embd=nn.Embedding(config.num_year,config.embedding_dim)
        
    
        #deep
        self.deep=nn.Sequential(
            nn.Linear(config.num_feature,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,2),  
        )

        self.config=config #this is used for testing
    
    def forward(self,product_id,user_id,year,month,day_of_week,hour,min_year,max_year):
        #lets insert some assertion here 
        assert (product_id >= 0).all() and (product_id<self.config.num_product).all(), "product id is out of bound"

        assert (user_id>=0).all() and (user_id<self.config.num_users).all(),"user id is out of bound"

        #the year is different
        assert (year>=min_year).all() and  (year<=max_year).all(),"year is out of bound" 

       

        assert (month>=0).all() and (month<self.config.num_month).all(),"month is out of bound"

        assert (day_of_week>=0).all() and (day_of_week<self.config.num_day_week).all(),"day of week is out of bound"

        assert (hour>=0).all() and (hour<=self.config.num_time_day).all(),"hour is out of bound"


        #emebedding the input
        product_id_embed=self.product_embed(product_id)
        user_id_embed=self.user_embed(user_id)
        year_embed=self.year_embd(year)
        time_month_embed=self.month_embed(month)
        day_week_embed=self.day_week_embed(day_of_week)
        time_day_embed=self.time_day_embed(hour)

        #feeding into wide
        wide_output=self.wide(torch.cat([product_id.float(),user_id.float()],dim=1))      #inputs are floats cause ouputs must be floats not torch long 

        #feeding into deep
        deep_input=torch.cat((
                product_id_embed,
                user_id_embed,
                year_embed,
                time_month_embed,
                day_week_embed,
                time_day_embed
        ))

        #output of deep
        deep_output=self.deep(deep_input)   
                              
        return torch.sigmoid(deep_output+wide_output)