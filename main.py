import torch
import torch.nn as nn
import warnings
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

        #lets initialize the embedding
        self.product_embed.weight.data.uniform_(-1,1)
        self.user_embed.weight.data.uniform_(-1,1)
        self.day_week_embed.weight.data.uniform_(-1,1)
        self.month_embed.weight.data.uniform_(-1,1)
        self.time_day_embed.weight.data.uniform_(-1,1)
        self.year_embd.weight.data.uniform_(-1,1)   
           
        #deep
        self.deep=nn.Sequential(
            nn.Linear(config.embedding_dim*6,1024), #config.num_feature
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512,216),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(216,200),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(200,128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128,80),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(80,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        

        #initializing the weights
        self._initialize_weights()

        self.config=config #this is used for testing
    
    def forward(self,product_id,user_id,year,month,day_of_week,hour):

        #lets insert some range of inputs 
        if not ((product_id >= 0).all() and (product_id<self.config.num_product).all()):
            warnings.warn("product id is out of bound")

        if not ((user_id>=0).all() and (user_id<self.config.num_users).all()):
            warnings.warn("user id is out of bound")

        if not ((month>=0).all() and (month<self.config.num_month).all()):
            warnings.warn("month is out of bound")

        if not ((day_of_week>=0).all() and (day_of_week<self.config.num_day_week).all()):
            warnings.warn("day of week is out of bound")


       #lets assert that inputs are not nan
        if torch.isnan(product_id).any():
            warnings.warn("there is a nan value in product id")

        if torch.isnan(user_id).any():
            warnings.warn("there is a nan value in user id")

        if torch.isnan(year).any():
            warnings.warn("there is a nan value in year")

        if torch.isnan(month).any():
            warnings.warn("thre is a nan value in month")

        if torch.isnan(day_of_week).any():
            warnings.warn("there is a nan value in day of week")

        if torch.isnan(hour).any():
            warnings.warn("there is a nan value in the hour")

        #emebedding the input
        product_id_embed=self.product_embed(product_id)
        user_id_embed=self.user_embed(user_id)
        year_embed=self.year_embd(year)
        time_month_embed=self.month_embed(month)
        day_week_embed=self.day_week_embed(day_of_week)
        time_day_embed=self.time_day_embed(hour)

        wide_output=self.wide(torch.cat((product_id.float(),user_id.float()),dim=1))      #inputs are floats cause ouputs must be floats not torch long 

        #before concatinating we have the following shape
        # print(f"product id embed: {product_id_embed.shape}")
        # print(f"user input embed: {user_id_embed.shape}")
        # print(f"year embed: {year_embed.shape}")
        # print(f"time of month: {time_month_embed.shape}")
        # print(f"day of week shape:{day_week_embed.shape}")
        # print(f"time of day embed: {time_day_embed.shape}")

        #feeding into deep but the shapes needs to be adjusted
        deep_input=torch.cat((
                product_id_embed.view(product_id_embed.size(0),-1),
                user_id_embed.view(user_id_embed.size(0),-1),
                year_embed.view(year_embed.size(0),-1),
                time_month_embed.view(time_month_embed.size(0),-1),
                day_week_embed.view(day_week_embed.size(0),-1),
                time_day_embed.view(time_day_embed.size(0),-1)
        ),dim=1)

        deep_output=self.deep(deep_input)   

        return deep_output+wide_output  #didn't include sigmoid because am using torch bce with logits as loss

    #lets initialize the model
    def _initialize_weights(self):

        for layer in self.deep:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.wide.weight)