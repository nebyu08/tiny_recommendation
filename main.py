#todo:
 #build the encoder of the :
    #build the wide model
    #build the deep model

import torch
import torch.nn as nn
from dataclasses import dataclass

#setting up the simple configuration
@dataclass
class WD_Config:
    num_product:int 
    num_users:int 
    num_day_week:int
    num_month:int 
    num_time_day:int
    num_feature:int
    embedding_dim:int

#main model
class WideDeep(nn.Module):
    """the implementation of the wide and deep neural network based on the paper:
            link:   https://arxiv.org/pdf/1606.07792v1
    """
    def __init__(self,config:WD_Config):   #I removed the "rate"
        super().__init__()

        #seetting up the config
       # self.config=config

        #wide
        self.wide=nn.Linear(2,1)    #2 features are product id and customer id

        #lets setup the embedding
        self.product_embed=nn.Embedding(config.num_product,config.embedding_dim)
        self.user_embed=nn.Embedding(config.num_users,config.embedding_dim)
        self.day_week_embed=nn.Embedding(config.num_day_week,config.embedding_dim)
        self.month_embed=nn.Embedding(config.num_month,config.embedding_dim)
        self.time_day=nn.Embedding(config.num_time_day,config.embedding_dim)
        
    
        #deep
        self.deep=nn.Sequential(
            nn.Linear(config.num_feature,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Softmax()
        )
    
    def forward(self,data):
        product_id_embed=self.product_embed(data['product_id'])
        user_id_embed=self.user_embed(data['user_id'])
        day_week_embed=self.day_week_embed(data['day_week'])
        time_month_embed=self.month_embed(data['month'])
        time_day_embed=self.time_day(data['time_day'])
        rate=data['rate']

        #feeding into wide
        wide_outptut=self.wide(data['user_id'],data['product_id'])

        #feeding into deep
        deep_input=torch.cat((
                product_id_embed,
                user_id_embed,
                day_week_embed,
                time_month_embed,
                time_day_embed,
                rate
        ))

        #output of deep
        deep_output=self.deep(deep_input)   
                              
        return torch.sigmoid(deep_output,wide_outptut)
    

#assigin the config of the model
config=WD_Config(
    num_product= 100
    num_users = 100
    num_day_week = 7
    num_month= 12
    num_time_day= 24
    num_feature =8
    embedding_dim=40
)


