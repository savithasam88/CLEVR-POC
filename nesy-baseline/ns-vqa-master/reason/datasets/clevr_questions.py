# CLEVR questions dataset
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils.utils as utils
from torchvision import transforms 
from PIL import Image

#cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#print(cuda_device)


class ClevrQuestionDataset(Dataset):

    def __init__(self, question_h5_path, max_samples, vocab_json, image_path):
        self.max_samples = max_samples
        question_h5 = h5py.File(question_h5_path, 'r')
        #print("Keys in h5 file:::")
        #print(question_h5.keys())
        

        self.questions = torch.LongTensor(np.asarray(question_h5['questions'], dtype=np.int64))
        self.image_idxs = question_h5['image_idxs']

        #self.image_idxs = np.asarray(question_h5['image_idxs'], dtype=np.int64)


        
        #self.complete_image = np.asarray(question_h5['complete_image_index'], dtype=np.int64)
        self.constraint_type = np.asarray(question_h5['constraint_type'], dtype=np.int64)
                
        self.image_path = image_path
        self.programs, self.answers = None, None
        
        
        if 'programs' in question_h5:
            self.programs = torch.LongTensor(np.asarray(question_h5['programs'], dtype=np.int64))
            
        if 'answers' in question_h5:
            self.answers = np.asarray(question_h5['answers'], dtype=np.int64)
        self.vocab = utils.load_vocab(vocab_json)
        
    def __len__(self):
        
        if self.max_samples:
            return min(self.max_samples, len(self.questions))
        else:
            return len(self.questions)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('index %d out of range (%d)' % (idx, len(self)))
        question = self.questions[idx]
        
        image_idx =self.image_idxs[idx]
        #complete_image = self.complete_image[idx]
        #l = len(str(complete_image))
        #print("Complete_idx:", complete_image)
        #print("Image idx:", image_idx)
        
        #img_path = os.path.join(self.image_path, image_filename)
        #img_path = os.path.normpath(img_path)
        #img = Image.open(img_path).convert('RGB')#img_path.convert("RGB")
        #convert_tensor = transforms.ToTensor()
        #img_tensor = convert_tensor(img)
        #print("Img_tensor shape:", img_tensor.shape)

        
        
        constraint_type = self.constraint_type[idx]
        
        program = -1
        answer = -1

        if self.programs is not None:
            program = self.programs[idx] 
        if self.answers is not None:
            answer = self.answers[idx]
        
        return question, program, answer, image_idx, constraint_type
