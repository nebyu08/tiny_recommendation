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
        self.layer=nn.Linear(n_inputs,n_outputs)

    def forward(self,inputs):   
        return self.layer(inputs)


#class Deep_Model(nn.Module):