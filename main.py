#todo:
 #build the encoder of the :
    #build the wide model
    #build the deep model

import torch
import torch.nn as nn

class WideDeep(nn.Module):
    """the implementation of the wide and deep neural network based on the paper:
            link:   https://arxiv.org/pdf/1606.07792v1
    """
    def __init__(self,num_product,num_users,rate,num_day_week,num_month,num_time_day,num_feature,embedding_dim=20):
        super().__init__()

        #wide
        self.wide=nn.Linear(2,1)    #2 features are product id and customer id

        #lets setup the embedding
        self.product_embed=nn.Embedding(num_product,embedding_dim)
        self.user_embed=nn.Embedding(num_users,embedding_dim)
        self.day_week_embed=nn.Embedding(num_day_week,embedding_dim)
        self.month_embed=nn.Embedding(num_month,embedding_dim)
        self.time_day=nn.Embedding(num_time_day,embedding_dim)
        
    
        #deep
        self.deep=nn.Sequential(
            nn.Linear(num_feature,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Softmax()
        )
    
    def forward(self,data):
        