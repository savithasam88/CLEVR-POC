import torch
import os
import pickle
from transformers import AutoTokenizer, AutoModel


def get_constraint_translation(env_folder, constraint_type_index, general_constraint_translation):
    sentences = []
    sentences.extend(general_constraint_translation)
    is_general = True
    constraints = open(os.path.join(env_folder, str(constraint_type_index) + '.lp'), 'r')
    # Using for loop
    
    for constraint in constraints:
        if '#show' in constraint:
            break
        if len(constraint.strip()) == 0:
            continue
        

        if not is_general:
            if '#count' in constraint:
                if 'sameProperty' in constraint:
                    
                    prop = constraint.split('sameProperty(X1, X2, ')[1].split(')')
                    prop_key = prop[0]
                    at_parts = constraint.split('at(')
                    region_index_1 = at_parts[1].split('X1, ')[1].split(')')[0]
                    region_index_2 = at_parts[2].split('X2, ')[1].split(')')[0]
                    
                    if '<' in constraint:
                        N = int(constraint.split('<')[1][0])/2
                        N = str(int(N))
                        limit = 'at least'
                        
                    else:
                        N = int(constraint.split('>=')[1][0])/2
                        N = str(int(N-1))
                        limit = 'at most'
                        
                        
                    if 'hasProperty' in constraint:
                        property_parts = constraint.split('hasProperty')
                        prop1 = property_parts[1].split('(X1, ')[1].split(',')
                        prop1_key = prop1[0]
                        prop1_value = prop1[1].split(')')[0].strip()
                        
                        sentences.append(f'There are {limit} {N} pairs of {prop1_key} {prop1_value} objects with same {prop_key} in regions {region_index_1} and {region_index_2} together.')                    
                        
                    else:
                        sentences.append(f'There are {limit} {N} pairs of objects with same {prop_key} in regions {region_index_1} and {region_index_2} together.')                    

                    
                        
                else:
                    N = constraint.split('!=')[1][0]
                    region_index = constraint.split('at(X, ')[1][0]
                    prop = constraint.split('hasProperty(X, ')[1].split(',')
                    prop_key = prop[0]
                    prop_value = prop[1].split(')')[0].strip()
                    sentences.append(f'There are exactly {N} {prop_value} {prop_key} objects in region {region_index}.')
                
            elif 'not' in constraint:
                region_index = constraint.split(':- object(X), at(X, ')[1][0]
                if constraint.count('not') == 1:
                    prop = constraint.split('hasProperty(X, ')[1].split(',')
                    prop_key = prop[0]
                    prop_value = prop[1].split(')')[0].strip()
                    sentences.append(f'All objects in region {region_index} have {prop_value} {prop_key}.')
                else:
                    not_parts = constraint.split('not')
                    prop1 = not_parts[1].split('hasProperty(X, ')[1].split(',')
                    prop1_key = prop1[0]
                    prop1_value = prop1[1].split(')')[0].strip()

                    prop2 = not_parts[2].split('hasProperty(X, ')[1].split(',')
                    prop2_key = prop2[0]
                    prop2_value = prop2[1].split(')')[0].strip()
                    sentences.append(f'All objects in region {region_index} have either {prop1_value} {prop1_key} or {prop2_value} {prop2_key}.')
                    
            else:
                if 'at' in constraint:
                    region_index = constraint.split(':- object(X), at(X, ')[1][0]
                    prop = constraint.split('hasProperty(X, ')[1].split(',')
                    prop_key = prop[0]
                    prop_value = prop[1].split(')')[0].strip()
                    sentences.append(f'There are no {prop_value} {prop_key} objects in region {region_index}.')
                else:
                    prop = constraint.split('hasProperty(X, ')[1].split(',')
                    prop_key = prop[0]
                    prop_value = prop[1].split(')')[0].strip()
                    sentences.append(f'There are no {prop_value} {prop_key} objects in the scene.')
        else:
            if 'object(0..' in constraint:
                obj_num = constraint.split('..')[1].split(')')[0]
                sentences.append(f'There are {obj_num} objects in the scene.')
                is_general = False
        
    with open(os.path.join(env_folder, str(constraint_type_index) + '.txt'), mode='w') as f1:
        f1.write('\n'.join(sentences))
    f1.close() 
    return sentences

                
                

def get_constraint_file(env_folder, constraint_type_index):
    sentences = []

    file1 = open(os.path.join(env_folder, str(constraint_type_index) + '.lp'), 'r')
    # Using for loop
    for line in file1:
        sentences.append(line.strip())
    return sentences

def get_general_constraint_file(data_folder):
    sentences = []

    file1 = open(os.path.join(data_folder, 'general_constraints_natural_language.txt'), 'r')
    # Using for loop
    for line in file1:
        sentences.append(line.strip())
    return sentences

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
           

def get_environment_embedding(env_folder, constraint_type_index, tokenizer, model, general_constraint_translation):
    
    #sentences = get_constraint_file(env_folder, constraint_type_index) 
    sentences = get_constraint_translation(env_folder, constraint_type_index, general_constraint_translation) 
    
    tokenizer.pad_token = tokenizer.eos_token


    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding = True, truncation=True, return_tensors='pt')
    #print(torch.size(encoded_input))
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


    # Perform pooling. In this case, mean pooling.
    lp_embeddings = pooled = torch.mean(sentence_embeddings, 0)#, encoded_input['attention_mask'])


    #from sentence_transformers import SentenceTransformer
    #sentences = [data]

    #model = SentenceTransformer('{MODEL_NAME}')
    #embeddings = model.encode(sentences)
    #print(embeddings)
    return lp_embeddings


def get_total_embedding(env_folder, data_folder):
    
    general_constraint_translation = get_general_constraint_file(data_folder)
    
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')    
    gpt2_model = AutoModel.from_pretrained('gpt2')

    total_embedding = {}
    all_file_names = [f for f in os.listdir(env_folder) if '.lp' in f]
    for f in all_file_names:
        print(f)         
        constraint_type_index = int(f.split('.')[0])
        total_embedding[constraint_type_index] = get_environment_embedding(env_folder, constraint_type_index, gpt2_tokenizer, gpt2_model, general_constraint_translation)         
    
    #with open(os.path.join(env_folder, 'total_embedding.pickle'), 'wb') as t:
    with open(os.path.join(env_folder, 'total_translation_embedding.pickle'), 'wb') as t:
        pickle.dump(total_embedding, t) 


if __name__ == "__main__":
    env_folder = '/home/code/CLEVR-POC/clevr-poc-dataset-gen/environment_constraints'
    data_folder = '/home/code/CLEVR-POC/clevr-poc-dataset-gen/data'
    get_total_embedding(env_folder, data_folder)
