from torchvision import models as models
import torch.nn as nn
from transformers import CLIPModel

def model(requires_grad,  
                 #checkpoint, 
                 clip_model,
                 clip_embedding_dim,
                 env_embedding_dim,
                 output_dim):
    
    
    #clip_model = CLIPModel.from_pretrained(checkpoint)
    for param in clip_model.parameters():
        param.requires_grad = requires_grad
    
    

        
    input_dim = clip_embedding_dim*2 + env_embedding_dim
    #input_dim = clip_embedding_dim*2


    classifier = nn.Linear(input_dim, output_dim) # load and initialize weights
       
    
    return clip_model, classifier