import torch
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy
from torchmetrics.classification import BinaryHammingDistance


def update(loss, targets):
    
    weights = torch.where(targets > 0, targets, 5)
    loss = torch.mul(loss, weights)
    loss = torch.mean(loss)
    return loss
    

def to_device(data, device):
        
        constraint_embedding = data['constraint_embedding'].to(device)
        target = data['target'].to(device)
        
        #input_ids = data['input_ids'].to(device)   
        #pixel_values = data['pixel_values'].to(device)   
        #attention_mask = data['attention_mask'].to(device)   

        return {
            'constraint_embedding': constraint_embedding, 
            'target': target,

            #'input_ids': input_ids,
            #'pixel_values': data['pixel_values'],
            #'attention_mask': data['attention_mask'],
            
            'image_path': data['image_path'],
            #'clip_input': data['clip_input'],
            'question': data['question'],
            'constraint_type': data['constraint_type'],
            'answer': data['answer']
        }

def get_pil_image(image_path):
    return Image.open(image_path).convert("RGB")  

def train(final_classifier, clip_model, dataloader, optimizer, criterion, train_data, device, dropout, clip_processor):
    print('Training')
    final_classifier.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        
        data_device = to_device(data, device)
       

        images = list(map(get_pil_image, data_device['image_path']))

        inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        clip_output = clip_model(**inputs)
        
        text_emb = clip_output['text_embeds']
        image_emb = clip_output['image_embeds']
        
        constraint_type_embedding = data_device['constraint_embedding']

        emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
        #emb = torch.cat([text_emb, image_emb], dim=1)          
        emb = dropout(emb)


        optimizer.zero_grad()
        
        outputs = final_classifier(emb)

        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
                
        #hamming_distance = BinaryHammingDistance().to(device)
        #loss = hamming_distance(outputs, data_device['target'].type(torch.int64))

        
        loss = criterion(outputs, data_device['target'])
        loss = update(loss, data_device['target'])

        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()


        
    train_loss = train_running_loss / counter
    return train_loss




# validation function
def validate(final_classifier, clip_model, dataloader, criterion, val_data, device, dropout, clip_processor, val_threshold):
    print('Validating')
    final_classifier.eval()
    counter = 0
    val_running_loss = 0.0
    val_running_acc = 0
    val_running_exact_acc = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            
            data_device = to_device(data, device)

            images = list(map(get_pil_image, data_device['image_path']))

            inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            clip_output = clip_model(**inputs)
            
            text_emb = clip_output['text_embeds']
            image_emb = clip_output['image_embeds']
            constraint_type_embedding = data_device['constraint_embedding']

            emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
            #emb = torch.cat([text_emb, image_emb], dim=1)          
            emb = dropout(emb)
            
            outputs = final_classifier(emb)

            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
        
            loss = criterion(outputs, data_device['target'])           
            loss = update(loss, data_device['target'])
            
            val_running_loss += loss.item()
            
            updated_outputs = torch.where(outputs > val_threshold, 1, 0.).tolist()[0]
            target = data_device['target'].tolist()[0]             
            
            #partial accuracy
            """
            r = numpy.where(numpy.array(target) == 1)
            ans_ones = set(list(r[0]))
            
            r = numpy.where(numpy.array(updated_outputs) == 1)
            pred_ones = set(list(r[0]))
            
            tp = len(list(ans_ones & pred_ones))
            fp = len(list(pred_ones.difference(ans_ones)))
            fn = len(list(ans_ones.difference(pred_ones)))
            jaccard_index = tp/(tp + fp + fn)
            val_running_acc += jaccard_index
            """
            
            #exact accuracy
            if numpy.array_equal(numpy.array(updated_outputs), numpy.array(target)):
                val_running_exact_acc += 1            
        
            
        val_loss = val_running_loss / counter
        #val_acc = val_running_acc / counter
        val_exact_acc = val_running_exact_acc /  counter
        return val_loss, val_exact_acc   
    



def test(final_classifier, clip_model, dataloader, criterion, test_data, device, dropout, clip_processor, val_threshold):
    print('Testing')
    final_classifier.eval()
    counter = 0
    test_running_partial_acc = 0
    test_running_exact_acc = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(test_data)/dataloader.batch_size)):
        counter += 1
        
        data_device = to_device(data, device)
        images = list(map(get_pil_image, data_device['image_path']))
       
        
        inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        clip_output = clip_model(**inputs)
            
        text_emb = clip_output['text_embeds']
        image_emb = clip_output['image_embeds']
        constraint_type_embedding = data_device['constraint_embedding']

        emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
        #emb = torch.cat([text_emb, image_emb], dim=1)          
        emb = dropout(emb)
          
        outputs = final_classifier(emb)

        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)  
        updated_outputs = torch.where(outputs > val_threshold, 1, 0.).tolist()[0]
        
        target = data_device['target'].tolist()[0]
        
        #partial accuracy
        #comm = numpy.sum(numpy.array(updated_outputs) == numpy.array(target))
        #test_running_partial_acc += comm/len(updated_outputs)
        
        r = numpy.where(numpy.array(target) == 1)
        ans_ones = set(list(r[0]))
           
        r = numpy.where(numpy.array(updated_outputs) == 1)
        pred_ones = set(list(r[0]))
            
        tp = len(list(ans_ones & pred_ones))
        fp = len(list(pred_ones.difference(ans_ones)))
        fn = len(list(ans_ones.difference(pred_ones)))
        jaccard_index = tp/(tp + fp + fn)        
        test_running_partial_acc += jaccard_index
        
        #exact accuracy
        if numpy.array_equal(numpy.array(updated_outputs), numpy.array(target)):
            test_running_exact_acc += 1
        
        
        #t = torch.max(data_device['target'], 1)[1]
        #correct += pred.eq(t).sum()
        
        

        """
        ## assign 0 label to those with less than 0.5
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0 

        N,C = data_device['target'].shape
        accuracy = (outputs == data_device['target']).sum() / (N*C) * 100   
        """
    
    #accuracy = 100 * correct / counter
    test_partial_acc = test_running_partial_acc /  counter
    test_exact_acc = test_running_exact_acc /  counter
    return test_exact_acc, test_partial_acc
