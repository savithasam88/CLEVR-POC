from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os, pickle
import json
import torch.nn as nn
import torch
import cv2
import numpy as np

from environment_embedding import get_environment_embedding
from transformers import CLIPProcessor

#Comment!!
clip_model_path = "openai/clip-vit-base-patch32"

extractor = CLIPProcessor.from_pretrained(clip_model_path)

def clip_transform_tokenize(question, image):
    return extractor(text=question,
                               images=image,
                               truncation=True, 
                               return_tensors="pt", 
                               #padding=True
                               padding="max_length", max_length=42
                               )


class ClevrPOCDataSet(Dataset):
    def __init__(self, data_folder, split, total_labels_to_index, env_folder):
        self.images_path = os.path.join(data_folder, 'images', split)
        self.questions_path = os.path.join(data_folder, 'questions', split)
        self.scenes_path = os.path.join(data_folder, 'scenes', split)
        self.total_labels_to_index = total_labels_to_index
        self.all_file_names = [f.split('.')[0] for f in os.listdir(self.images_path) if os.path.isfile(os.path.join(self.images_path, f))]

        self.env_folder = env_folder

        #with open(os.path.join(env_folder, 'total_embedding.pickle'), 'rb') as f:
        with open(os.path.join(env_folder, 'total_translation_embedding.pickle'), 'rb') as f:
            self.total_embedding = pickle.load(f)        

        
        
            
        
    def __len__(self):
        return len(self.all_file_names)
    
    def __getitem__(self, index):
        
              
        image_path = os.path.join(self.images_path, self.all_file_names[index] + '.png')
        #image = Image.open(image_path).convert("RGB")

        
        question_path = os.path.join(self.questions_path, self.all_file_names[index] + '.json')
        with open (question_path, 'rb') as f:
            question_dict = json.load(f)
            
        scene_path = os.path.join(self.scenes_path, self.all_file_names[index] + '.json')
        with open (scene_path, 'rb') as f:
            scene_dict = json.load(f)


        question = question_dict['question'] 

        #clip_input = clip_transform_tokenize(question, image)
        #pixel_values = clip_input['pixel_values']
        #attention_mask = clip_input['attention_mask']
        #input_ids = clip_input['input_ids']

        constraint_type_index = int(scene_dict['constraint_type_index'])

        #constraint_embedding = get_environment_embedding(self.env_folder, constraint_type_index, self.gpt2_tokenizer, self.gpt2_model)
        constraint_embedding = self.total_embedding[constraint_type_index]
        
        #answer = question_dict['answer'].replace('\"', '').strip('][').split(', ')
        #
        answer = question_dict['answer']
        a = [self.total_labels_to_index[i] for i in answer]
        b = [1 if i in a else 0 for i in range(len(self.total_labels_to_index))]       
        target = torch.Tensor(b) 
        


        return {
            'image_path': image_path,
            'question': str(question), 
            'constraint_type': torch.tensor(constraint_type_index, dtype=torch.int32),
            'constraint_embedding': constraint_embedding,
            'answer': '*'.join(answer),
            'target': target
            #'pixel_values': pixel_values,
            #'attention_mask': attention_mask,
            #'input_ids': torch.tensor(input_ids),
            #'clip_input': clip_input
        }
