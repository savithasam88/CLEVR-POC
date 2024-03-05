import os
import json
import numpy as np
import torch


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def invert_dict(d):
  return {v: k for k, v in d.items()}
  

def load_vocab(path):
    #print("Loading vocab from:",path)
    with open(path, 'r') as f:
        
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['labels_idx_to_token'] = invert_dict(vocab['labels'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2

    return vocab

def getNamesComb(f1, f2, vals):
    val_names = []
    COLORS = {'0':'blue', '1':'brown', '2':'cyan', '3':'gray', '4':'green', '5':'purple', '6':'red', '7':'yellow'}
    MATERIALS = {'0':'rubber', '1':'metal'}
    SHAPES = {'0':'cube', '1':'cylinder', '2':'sphere'}
    SIZES = {'0':'large', '1':'small'}
    
    map_dict1 = {}
    if f1 == "color":
        map_dict1 = COLORS.copy()
    elif f1 == "shape":
        map_dict1 = SHAPES.copy()
    elif f1 == "material":
        map_dict1 = MATERIALS.copy()
    elif f1 == "size":
        map_dict1 = SIZES.copy()
    
    map_dict2 = {}
    if f2 == "color":
        map_dict2 = COLORS.copy()
    elif f2 == "shape":
        map_dict2 = SHAPES.copy()
    elif f2 == "material":
        map_dict2 = MATERIALS.copy()
    elif f2 == "size":
        map_dict2 = SIZES.copy()
    
    for vl in vals:
        val_names.append([map_dict1[str(vl[0])],map_dict2[str(vl[1])]])
    return val_names

def getNames(feature, vals):
    
    val_names = []
    COLORS = {'0':'blue', '1':'brown', '2':'cyan', '3':'gray', '4':'green', '5':'purple', '6':'red', '7':'yellow'}
    MATERIALS = {'0':'rubber', '1':'metal'}
    SHAPES = {'0':'cube', '1':'cylinder', '2':'sphere'}
    SIZES = {'0':'large', '1':'small'}
    
    map_dict = {}
    if feature == "color":
        map_dict = COLORS.copy()
    elif feature == "shape":
        map_dict = SHAPES.copy()
    elif feature == "material":
        map_dict = MATERIALS.copy()
    elif feature == "size":
        map_dict = SIZES.copy()
    for v in vals:
        val_names.append(map_dict[str(v)])
    
    return val_names

def load_constraints(constraints_json):
    print("Constraints file:", constraints_json)
    
    
    with open(constraints_json) as f:
        constraints_dict = json.load(f)['constraints_set']
        #scenes_dict = json.load(f)['scenes']
    for s in constraints_dict:# for each constraint
        #list of constraints - each a dictionary with 9 keys -  regions 
        
        constraints = []
        for c in s:
            
            regions = {}
            list_constraints = s[c]['regions']
            count = 0
            for r in list_constraints:
                reg_cons = {}
                region_constraints = r['constraints'] #is a dictionary
                #print(region_constraints)
                for feature in region_constraints:
                    vals = region_constraints[feature]
#                    f1 = ""
#                    f2 = ""
#                    if "_" in feature:
#                        f1 = feature.split('_')[0]
#                        f2 = feature.split('_')[1]
#                    if (f1!="" and f2!=""):
#                        vals_name = getNamesComb(f1,f2, vals)
#                    else: 
#                        vals_name = getNames(feature, vals)
                    reg_cons[feature] = vals #vals_name
                regions[count]= reg_cons
                count=count+1
            constraints.append(regions)
    
    #print("Constraint-4:", constraints[4])    
    #print("Constraint-0:", constraints[0])               
    return constraints



def load_scenes(scenes_json):
    flag = 0
    if ('complete' in scenes_json):
        flag= 1
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    ind = 0
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                #item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                #                    np.dot(o['3d_coords'], s['directions']['front']),
                #                    o['3d_coords'][2]]
                item['position'] = o['3d_coords']
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            if flag ==0:
            	item['query'] = s['object_of_interest_index']
            table.append(item)
        scenes.append(table)
        
    return scenes
    

def load_embedding(path):
    return torch.Tensor(np.load(path))
