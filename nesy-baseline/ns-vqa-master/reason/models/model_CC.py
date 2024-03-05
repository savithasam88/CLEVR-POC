import torch
from torchvision import transforms 
from torchvision import models as models
import torch.nn as nn

class ImageConstraintTypeClassification(nn.Module):
    def __init__(self, device, num_constraint_types, opt):
        super(ImageConstraintTypeClassification,self).__init__()    
        self.device = device
        
        self.model = models.resnet50(progress=True, pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(2048, num_constraint_types)
        self.model.to(self.device)
        self.gpu_ids = opt.gpu_ids
    
    def forward(self, x):
        self.output_logprobs = self.model(x)
        v, self.predicted_constraint_type = torch.max(self.output_logprobs.data, 1)
        return self.predicted_constraint_type

    def backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None, 'must call reinforce_forward first'
        grad_output = None  
        
        loss = - torch.diag(torch.index_select(self.output_logprobs, 1, self.predicted_constraint_type)).sum()*reward \
                       + entropy_factor*(self.output_logprobs*torch.exp(self.output_logprobs)).sum()
        
        loss_sum = loss.sum()
        torch.autograd.backward(loss.sum(), grad_output, retain_graph=True)
        return loss.sum()

   
   
