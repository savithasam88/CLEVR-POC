import pickle
import json

properties = {
  "shape": {
    "cube": "SmoothCube_v2",
    "sphere": "Sphere",
    "cylinder": "SmoothCylinder",
    "cone": "Cone"
  },
  "color": {
    "gray": [87, 87, 87],
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51],
    "coral": [255, 127, 80]
  },
  "material": {
    "rubber": "Rubber",
    "metal": "MyMetal"
  },
  "size": {
    "large": 0.9,
    "medium": 0.6,
    "small": 0.3
  }
}

#constraints_pickle = pickle.load(open("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/data/raw/CLEVR_v1.0/scenes/constraint_types_tensor_30.pickle", "rb"))
#print(constraints_pickle)
#constraints_json = {}
#constraints_json["constraints_set"] = [constraints_pickle] 
constraints_json = "/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/data/raw/CLEVR_v1.0/scenes/constraints.json"
#region_constraint = pickle.load(open( "/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/executors/consistent_combination.p", "rb" ))

#Reading Constraints

consistent_combinations = {}
with open(constraints_json) as f:
  constraints_dict = json.load(f)['constraints_set']
  for s in constraints_dict:# for each constraint
  #list of constraints - each a dictionary with 9 keys -  regions 
    constraints = []
    for c in s:
      regions = {}
      list_constraints = s[c]['regions']
      count = 0
      #print("Constraint:", c)
      for r in list_constraints:
        reg_cons = {}
        region_constraints = r['constraints'] #is a dictionary
        cons_list = []
        for comb in region_constraints:
          l = []
          l.append(comb['color'])  
          l.append(comb['material'])
          l.append(comb['shape'])  
          l.append(comb['size'])
          l.append(count)
          cons_list.append(l)
        consistent_combinations[(int(c.split("_")[2]), count)] = cons_list
        count=count+1
print(consistent_combinations)
pickle.dump(consistent_combinations, open("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/consistent_combination.p", "wb" ) )



a = ["color", "material", "shape", "size"]
#key = (cid, feature, val) --> list of regions where feature = val

region_constraint = consistent_combinations
dict_cfv = {} 
for (cid, rid) in region_constraint:
  consis = region_constraint[(cid, rid)]
  for f in properties:
    ind = a.index(f)
    values = properties[f]
    for v in values:
      for c in consis:
        if c[ind] == v:
          if (cid, f, v) not in dict_cfv:
            dict_cfv[(cid, f, v)] = [rid]
          else:
            if rid not in dict_cfv[(cid, f, v)]:
              dict_cfv[(cid, f, v)].append(rid)
          break    

print("Dict_cfv::")
print(dict_cfv)
pickle.dump(dict_cfv, open("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/dict_cfv.p", "wb" ) )

dict_crf = {}
for (cid, rid) in region_constraint:
  consis = region_constraint[(cid, rid)]
  for f in properties:
    ind = a.index(f)
    for c in consis:
      if (cid, rid, f) not in dict_crf:
        dict_crf[(cid, rid, f)] = [c[ind]]
      else:
        if c[ind] not in dict_crf[(cid, rid, f)]:
              dict_crf[(cid, rid, f)].append(c[ind]) 
print("Dict_crf::")
print(dict_crf)

pickle.dump(dict_crf, open("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/dict_crf.p", "wb" ) )
