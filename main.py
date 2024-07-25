#todo:
 #build the encoder of the :
    #build the wide model
    #build the deep model

import torch
import torch.nn as nn


class Wide_Model(nn.Module):
    '''
    this the wide part of the model
    '''
    def __init__(self,n_inputs,n_outputs):
        super().__init__()
        self.layer=nn.Linear(n_inputs,n_outputs)

    def forward(self,anime_id,genre,type,episodes,general_rating,members,user_id,user_rating):   
        return self.layer(torch.cat(anime_id,genre,type,episodes,general_rating,members,user_id,user_rating))


class Deep_Model(nn.Module):
    """this make a deep neural network for the recommendation system

    Args:
        nn (module): for using torchs graph for forward and backward pass
    """
    def __init__(self,genre):
        super().__init__()
        
        #lets setup the embedding
        self.embedding=nn.Embedding(genre,torch.ceil(torch.sqrt(genre)))

        
        # self.n_episode=torch.tensor(n_episodes)
        # self.rating=torch.tenosr(rating)
        # self.members=torch.tensor(members)

        #setting up the neural netowrk part of the model
        self.layers=nn.Sequential(
            nn.Linear(7,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Softmax()
        )
    
    def forward(self,anime_id,genre,type,episodes,general_rating,members,user_id,user_rating):
        #lets insert the categorical into numeric
        embeding=self.embedding(genre)
        total_inputs=torch.cat(anime_id,embeding,type,episodes,general_rating,members,user_id,user_rating)
        return total_inputs
    
    class main_model(nn.Module):
        def __init__(self,n_inputs,n_outputs,genre):
            super().__init__()
            self.wide=Wide_Model(n_inputs,n_outputs)  #initialize the model
            self.deep=Deep_Model(genre)
        
        def forward(self,a):