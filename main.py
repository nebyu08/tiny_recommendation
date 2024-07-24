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

    def forward(self,inputs):   
        return self.layer(inputs)


class Deep_Model(nn.Module):
    """this make a deep neural network for the recommendation system

    Args:
        nn (module): for using torchs graph for forward and backward pass
    """
    def __init__(self,
                 anime_id,
                 user_id,
                 user_rate,
                 num_genre,
                 num_type_show,
                 n_episodes,
                 rating,
                 members
                 ):
        
        #lets setup the embedding
        self.embedding=nn.Embedding(num_genre,torch.ceil(torch.sqrt(num_genre)))

        
        self.n_episode=torch.tensor(n_episodes)
        self.rating=torch.tenosr(rating)
        self.members=torch.tensor(members)


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
        
        pass