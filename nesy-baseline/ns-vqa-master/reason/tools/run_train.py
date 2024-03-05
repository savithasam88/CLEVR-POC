import sys, os
import torch
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).parents[1], 'options'))
sys.path.append(os.path.join(Path(__file__).parents[1]))
#sys.path.append("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/options") # Adds higher directory to python modules path.
#sys.path.append("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason")
#sys.path.append("/home/Documents/ns-vqa-master/reason/executors")
#sys.path.append("/home/Documents/ns-vqa-master/reason/models")

#/home/Documents/ns-vqa-master/reason/options
from train_options import TrainOptions
from datasets import get_dataloader
#from executors import get_executor
from models.parser import Seq2seqParser
from trainer import Trainer



import warnings

if torch.cuda.is_available():
  print("cuda available..")  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
warnings.filterwarnings("ignore")
    
opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt).to(device = device)

#executor = get_executor(opt)
trainer = Trainer(opt, train_loader, val_loader, model,  device)

trainer.train()
