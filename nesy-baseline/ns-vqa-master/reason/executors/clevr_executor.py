import random
import json, sys
import utils.utils as utils
import pandas as pd
import itertools
import numpy as np
import pickle
import warnings




warnings.filterwarnings("ignore")

CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow', 'coral']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere', 'cone']
CLEVR_SIZES = ['large', 'small', 'medium']

##REGIONS = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal_color': ['yes', 'no'],
    'equal_integer': ['yes', 'no'],
    'equal_material': ['yes', 'no'],
    'equal_shape': ['yes', 'no'],
    'equal_size': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query_color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow', 'coral'],
    'query_material': ['metal', 'rubber'],
    'query_size': ['small', 'large', 'medium'],
    'query_shape': ['cube', 'cylinder', 'sphere', 'cone' ],
    'same_color': ['yes', 'no'],
    'same_material': ['yes', 'no'],
    'same_size': ['yes', 'no'],
    'same_shape': ['yes', 'no']
}
region_rel = {}
region_rel[0] = [[0,1,2],[0,1,2,3,4,5,6,7,8],[0,3,6],[0,1,2,3,4,5,6,7,8]]
region_rel[1] = [[0,1,2],[0,1,2,3,4,5,6,7,8],[0,1,3,4,6,7],[1,2,4,5,7,8] ]
region_rel[2] = [[0,1,2],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[2,5,8]]
region_rel[3] = [[0,1,2,3,4,5],[3,4,5,6,7,8],[0,3,6],[0,1,2,3,4,5,6,7,8]]
region_rel[4] = [[0,1,2,3,4,5], [3,4,5,6,7,8],[0,1,3,4,6,7],[1,2,4,5,7,8]]
region_rel[5] = [[0,1,2,3,4,5],[3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[2,5,8]]
region_rel[6] = [[0,1,2,3,4,5,6,7,8],[6,7,8],[0,3,6],[0,1,2,3,4,5,6,7,8]]
region_rel[7] = [[0,1,2,3,4,5,6,7,8],[6,7,8],[0,1,3,4,6,7], [1,2,4,5,7,8]]
region_rel[8] = [[0,1,2,3,4,5,6,7,8],[6,7,8],[0,1,2,3,4,5,6,7,8],[2,5,8]]

class ClevrExecutor:
    """Symbolic program executor for CLEVR"""
    def __init__(self, train_scene_json, val_scene_json, vocab_json, complete_train_scenes_json, complete_val_scenes_json, constraints_json):
    ##def __init__(self, train_scene_json, val_scene_json, vocab_json):
        self.scenes = {
            'train': utils.load_scenes(train_scene_json),
            'val': utils.load_scenes(val_scene_json),
            'complete-train':utils.load_scenes(complete_train_scenes_json),
            'complete-val':utils.load_scenes(complete_val_scenes_json),
            #'complete-test':utils.load_scenes(complete_val_scenes_json),
            #'constraint':utils.load_constraints(constraints_json)
        }
        #print(self.scenes['complete'][0])
        self.vocab = utils.load_vocab(vocab_json)
        self.colors = CLEVR_COLORS
        self.materials = CLEVR_MATERIALS
        self.shapes = CLEVR_SHAPES
        self.sizes = CLEVR_SIZES
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES
        self.region_rel = region_rel
        self.consistent = pickle.load(open( "/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/consistent_combination.p", "rb" ))
        self.cons_feat_val = pickle.load(open( "/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/dict_cfv.p", "rb" ))
        self.cons_reg_feat = pickle.load(open( "/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason/executors/dict_crf.p", "rb" ))
        self.modules = {}
        self._register_modules()
    
    def findRegionsRel(self, region_prev, rel):
        
        
        region_ret = []
       
        if rel == 'right':
            for r in region_prev:
                region_ret.extend(self.region_rel[r][3])
                
        elif rel == 'left':
            for r in region_prev:
              region_ret.extend(self.region_rel[r][2])
                
        elif rel == 'front':
            for r in region_prev:
              region_ret.extend(self.region_rel[r][1])
                
        elif rel == 'behind':
            for r in region_prev:
                region_ret.extend(self.region_rel[r][0])
                
        return region_ret                                
                
    
    def intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))
    
    ##Returns regions where feature has the value 
    def getRegions_feature(self, constraint, feature, value):
        return self.cons_feat_val[(constraint, feature, value)]
        """
        feature_regions = []
        
        for r in constraint:
            reg_cons = constraint[r]
            if feature in reg_cons:
                
                if value in reg_cons[feature]:
                    feature_regions.append(r)
                    
            else:
                for keys in reg_cons:
                    if ('_') in keys:
                        f1 = keys.split('_')[0]
                        f2 = keys.split('_')[1]
                        if f1 == feature:
                            for l in reg_cons[keys]:
                                if value == l[0]:
                                    feature_regions.append(r)
                        elif f2 == feature:
                            for l in reg_cons[keys]:
                                if value == l[1]:
                                    feature_regions.append(r)
        
        return feature_regions
        """                        
                
    
    
    def getRegions(self, scene, constraint):
        object_regions = {}
        
        for o in scene:
            color = []
            material = []
            shape = []
            size = []
            for r in constraint:
                reg_cons = constraint[r]
                if 'color' in reg_cons:
                    if o['color'] in reg_cons['color']:
                        color.append(r)
                if 'shape' in reg_cons:
                    if o['shape'] in reg_cons['shape']:
                        shape.append(r)
                if 'material' in reg_cons:
                    if o['material'] in reg_cons['material']:
                        material.append(r)
                if 'size' in reg_cons:
                    if o['size'] in reg_cons['size']:
                        size.append(r)
                if 'color_size' in reg_cons:
                    if [o['color'], o['size']] in reg_cons['color_size']:
                        color.append(r)
                        size.append(r)
                if 'color_shape' in reg_cons:
                    if [o['color'], o['shape']] in reg_cons['color_shape']:
                        color.append(r)
                        shape.append(r)
                if 'color_material' in reg_cons:
                    if [o['color'], o['material']] in reg_cons['color_material']:
                        color.append(r)
                        material.append(r)
                if 'material_size' in reg_cons:
                    if [o['material'], o['size']] in reg_cons['material_size']:
                        size.append(r)
                        material.append(r)
                if 'shape_size' in reg_cons:
                    if [o['shape'], o['size']] in reg_cons['shape_size']:
                        size.append(r)
                        shape.append(r)
            cm = self.intersection(color, material)
            cms = self.intersection(shape, cm)
            cmss = self.intersection(size, cms)
            object_regions[o] = cmss
        return object_regions

    def getFeatureVals_Region(self, region, constraint, feature):
        return self.cons_reg_feat[(constraint, region, feature)]
        """
        feature_vals = []
        for r in constraint:
            if r!=region:
                continue
            reg_cons = constraint[r]
            if feature in reg_cons:
                for value in reg_cons[feature]:
                    feature_vals.append(value)
                break
            else:
                for keys in reg_cons:
                    if ('_') in keys:
                        f1 = keys.split('_')[0]
                        f2 = keys.split('_')[1]
                        if f1 == feature:
                            for l in reg_cons[keys]:
                                feature_vals.append(l[0])
                        elif f2 == feature:
                            for l in reg_cons[keys]:
                                feature_vals.append(l[1])
                break
        return feature_vals
        """
    def getConsistent(self, region, constraint):
    
        
        return self.consistent[(constraint, region)]#pickle.load(open( "/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/executors/consistent_combination.p", "rb" ))
        
        """
        comb = {}
        
        colors = self.getFeatureVals_Region(region, constraint, 'color')
        materials = self.getFeatureVals_Region(region, constraint, 'material')
        shape = self.getFeatureVals_Region(region, constraint, 'shape')
        size = self.getFeatureVals_Region(region, constraint, 'size')
        
        dict_code = {'color':0, 'material':1, 'shape':2, 'size':3}
        for r in constraint:
            if r!=region:
                continue
            reg_cons = constraint[r]
            for keys in reg_cons:
                if "_" in keys:
                    f1 = keys.split('_')[0]
                    f2 = keys.split('_')[1]
                    comb[(dict_code[f1],dict_code[f2])] = reg_cons[keys]
                
        a = [colors, materials, shape, size]
        comb_list = list(itertools.product(*a))
        consistent = []
        for l in comb_list:
            flag = 0
            for i in range(len(l)-1):
                for j in range(i+1,len(l)):
                    if (i,j) in comb:
                        if [l[i],l[j]] not in comb[(i,j)]:
                            flag = 1
                            break
                if flag == 1:
                    break
            if flag == 0:
                l = list(l)
                l.append(region)
                if l not in consistent:
                    consistent.append(l)
        """
        #consistent = [['blue', 'rubber', 'cube', 'small', 1], ['blue', 'rubber', 'cube', 'small', 2]]
        
            
    def compute_entropy(self, feature):
        probs = feature.value_counts(normalize=True)
        impurity = -1 * np.sum(np.log2(probs) * probs)
        return round(impurity, 3)
    
    def compute_information_gain(self,df, target, descriptive_feature):
        #print('target feature:', target)
        #print('feature_value:', descriptive_feature)
    
        target_entropy = self.compute_entropy(feature=df[target])
        entropy_list = list()
        weight_list = list()

        for feature_value in df[descriptive_feature].unique():
            df_feature_value = df[df[descriptive_feature] == feature_value]
            entropy_feature_value = self.compute_entropy(df_feature_value[target])
            entropy_list.append(round(entropy_feature_value, 3))
            weight_feature_value = len(df_feature_value) / len(df)
            weight_list.append(round(weight_feature_value, 3))

        feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))   
        information_gain = target_entropy - feature_remaining_impurity

        return information_gain
    
    def find_best_feature(self, df, target, query_feature):
        information_gain_values = {}
        for feature in df.drop([target, query_feature], axis = 1).columns:
            
            information_gain_values[feature] = self.compute_information_gain(df, target, feature)
        best_feature = max(information_gain_values, key=information_gain_values.get)
        return best_feature

    def find_best_feature_U(self, dict_poss, target, query_feature):
        maxi = 0
        feat = ''
        for key in dict_poss:
            if key == query_feature or key == target:
                continue
            num_unique = len(set(dict_poss[key]))
            if num_unique>maxi:
                feat = key
                maxi = num_unique
        return feat
    
    def query_feature(self, obj, scene, feature):
        if type(obj) == dict and feature in obj:
            return obj[feature]
        
        return 'error'
    
    def getQuery_Object(self, scene, complete_scene):
        scene = list(scene)
        query_obj_id = scene[0]['query']
        complete_scene = list(complete_scene)
        query_obj = complete_scene[query_obj_id]
        
        return query_obj
    
    def reason_U(self, ct_type, complete_scene, query_obj, region_seq, token_seq, outputs_seq):
        #print("Token_Seq:", token_seq)
        #print("Region_Seq:", region_seq)
        #print("Constraint_type:", ct_type)
        #print("Complete_scene:", complete_scene)
        query_obj_regions = region_seq[len(region_seq)-1]
        #print("QOR:", query_obj_regions)
        data = []
        for r in query_obj_regions:
            consistent_combs = self.getConsistent(r, ct_type)
            data.extend(consistent_combs)
        
        if ('_' not in token_seq[len(token_seq)-1]):
          #print("Returning error as _ not:", token_seq )
          return 'error'
        query_feature = token_seq[len(token_seq)-1].split("_")[1]
        query_feature = str(query_feature.strip())
        #print("Query_feature:", query_feature)
        
        dict_poss = {'color':[], 'material':[], 'shape':[], 'size':[], 'region':[]}
        
        if(len(data) == 0):
            #print("Returning error as len(data) is 0")
            return 'error'
        
        for l in data:
            dict_poss['color'].append(l[0])
            dict_poss['material'].append(l[1])
            dict_poss['shape'].append(l[2])
            dict_poss['size'].append(l[3])
            dict_poss['region'].append(l[4])
        #print("Dict_poss-1:", dict_poss)
        if len(dict_poss[query_feature]) == 1:
            ans = dict_poss[query_feature][0]
            
        feature_req_prev = ""
        ans = -1
        while(ans==-1):
            feature_req = self.find_best_feature_U(dict_poss,  target = 'region', query_feature=query_feature)
            feature_req_val = self.query_feature(query_obj, complete_scene, feature_req)
            if (feature_req_prev==feature_req):
              #print("Returning error as fet_prev,fet_cur")
              return 'error'
            feature_req_prev = feature_req
            #print("F, FV:", feature_req, feature_req_val)
            indices = [i for i, x in enumerate(dict_poss[feature_req]) if x == feature_req_val]
            for key in dict_poss:
                dict_poss[key] = [elem for i,elem in enumerate(dict_poss[key]) if i in indices]
            #print("After update:",dict_poss)
            if len(dict_poss['color'])==0:
                #print("Retrning error as df is empty:", dict_poss)
                return 'error'
                
            #print(df)
            
            if (len(set(dict_poss[query_feature])) == 1):
                ans = dict_poss[query_feature][0]

        #print("Returning ans:", ans)
        return ans
          
    def reason(self, constraint, ct_type, complete_scene, query_obj, region_seq, token_seq, outputs_seq):
        ##get the last set of possible regions
        
        #print("Possible regions of query object:", query_obj,"::", region_seq[len(region_seq)-1])
        
        ##Loop information gain - get feature to question --> query complete_scene --> get answer --> update dataset --> check for answer
        query_obj_regions = region_seq[len(region_seq)-1]
        data = []
        for r in query_obj_regions:
            consistent_combs = self.getConsistent(r, ct_type)
            data.extend(consistent_combs)
        if ('_' not in token_seq[len(token_seq)-1]):
          return 'error'
        query_feature = token_seq[len(token_seq)-1].split("_")[1]
        query_feature = str(query_feature.strip())
        #print("Query_feature:", query_feature)
        df = pd.DataFrame(data, columns = ['color', 'material', 'shape', 'size', 'region'])
        if df.empty:
            return 'error'
        ans = -1
        
        #print("Not empty initially..")
        #print(df)
        if (df[query_feature] == df[query_feature][0]).all():
                ans = df[query_feature][0]
        feature_req_prev = ""

        while(ans==-1):
            feature_req = self.find_best_feature(df=df, target='region', query_feature=query_feature)
            feature_req_val = self.query_feature(query_obj, complete_scene, feature_req)
            #print("Feature_req:", feature_req," :: ", feature_req_val)
            if (feature_req_prev==feature_req):
            	return 'error'
            feature_req_prev = feature_req
            df = df.drop(df[df[feature_req] != feature_req_val].index)
            df = df.reset_index(drop = True)
            if df.empty:
                return 'error'
                
            #print(df)
            
            if (df[query_feature] == df[query_feature][0]).all():
                ans = df[query_feature][0]

        return ans  
                        
    def run(self, x, index, ci_index, ct_type, split, guess=False, debug=False):
    ##def run(self, x, index, split, guess=False, debug=False):
        
        
        assert self.modules and self.scenes, 'Must have scene annotations and define modules first'
        assert split == 'train' or split == 'val'
        flag_reason = 0
        flag_error = 0
        flag_re = 0
        flag_dir = 0
        g = 0
        ans, temp = None, None
        region_ans, region_temp = None, None
        
        token_seq = []
        outputs_seq = []
        region_seq = []
        #print("Vocab of program_idx_token")
        #print(self.vocab['program_idx_to_token'])
        
        ##token_seqt = ['scene', 'filter_size[small]', 'filter_color[brown]', 'unique', 'same_shape', 'filter_material[metal]', 'unique', 'query_size']
        
        # Find the length of the program sequence before the '<END>' token
        length = 0
        #print("X::::::",x)
        #count_key = 0 
        for k in range(len(x)):
            l = len(x) - k
            if x[l-1] not in self.vocab['program_idx_to_token']:
              flag_error = 1
              return 'error', flag_reason, flag_error, flag_re, flag_dir,g   
            if self.vocab['program_idx_to_token'][x[l-1]] == '<END>':
                length = l
        if length == 0:
            flag_error = 1
            return 'error', flag_reason, flag_error, flag_re, flag_dir,g 
        
        scene = self.scenes[split][index]
        complete_scene = self.scenes['complete-'+split][ci_index]
        query_obj = self.getQuery_Object(scene, complete_scene)
        #constraint = self.scenes['constraint'][ct_type]
        
        #print("Complete_scene:::", complete_scene)
        #print("Constraint type:", constraint, ct_type)
        #obj_regions = self.getRegions(self, scene, constraint)
        
        self.exe_trace = []
        ##t_c =0
        ##length = 8
        for j in range(length):
            i = length - 1 - j
            token = self.vocab['program_idx_to_token'][x[i]]
            ##token = token_seqt[t_c]
            ##t_c= t_c+1
            token_seq.append(token)
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                
                temp = ans
                region_temp = region_ans
                ans = list(scene)
                region_ans = [0,1,2,3,4,5,6,7,8] #set of all regions
                region_seq.append(region_ans)
                outputs_seq.append(ans)
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans, region_ans = module(ans, ct_type, region_ans, scene,[0,1,2,3,4,5,6,7,8])
                else:
                    ans, region_ans = module(ans, ct_type, region_ans, temp, region_temp)
                
                region_seq.append(region_ans)
                outputs_seq.append(ans)
                if ans == 'error':
                    break
            self.exe_trace.append(ans)
            
            if debug:
                self._print_debug_message(ans)
                self._print_debug_message(temp)
                print()
        ans = str(ans)
        #for tok in token_seq:
           #if ('same_' in tok or 'relate' in tok):
               #print("Before reason - Program generated:", token_seq)
               #print("Before reason - Regions:", region_seq)
               #break
        
        if ans == 'error':
            flag_error = 1 
        elif ans == str(-1):
             #print("Ans is empty...")
             ans = self.reason_U(ct_type, complete_scene, query_obj, region_seq, token_seq, outputs_seq)
             flag_reason =1
             if ans == 'error':
               flag_re = 1 
        else:
          flag_dir = 1
        
        if ans == 'error' and guess:
          final_module = self.vocab['program_idx_to_token'][x[0]]
          if final_module in self.answer_candidates:
              ans = random.choice(self.answer_candidates[final_module])
              g = 1
        return ans, flag_reason, flag_error, flag_re, flag_dir, g

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        return '%s %s %s %s at %s' % (obj['size'], obj['color'], obj['material'], obj['shape'], str(obj['position']))
    
    def _register_modules(self):
#        self.modules['count'] = self.count
#        self.modules['equal_color'] = self.equal_color
#        self.modules['equal_integer'] = self.equal_integer
#        self.modules['equal_material'] = self.equal_material
#        self.modules['equal_shape'] = self.equal_shape
#        self.modules['equal_size'] = self.equal_size
        #self.modules['exist'] = self.exist
       
        self.modules['filter_color[blue]'] = self.filter_blue
        self.modules['filter_color[brown]'] = self.filter_brown
        self.modules['filter_color[cyan]'] = self.filter_cyan
        self.modules['filter_color[gray]'] = self.filter_gray
        self.modules['filter_color[green]'] = self.filter_green
        self.modules['filter_color[purple]'] = self.filter_purple
        self.modules['filter_color[red]'] = self.filter_red
        self.modules['filter_color[yellow]'] = self.filter_yellow
        self.modules['filter_color[coral]'] = self.filter_coral
        self.modules['filter_material[rubber]'] = self.filter_rubber
        self.modules['filter_material[metal]'] = self.filter_metal
        self.modules['filter_shape[cube]'] = self.filter_cube
        self.modules['filter_shape[cylinder]'] = self.filter_cylinder
        self.modules['filter_shape[sphere]'] = self.filter_sphere
        self.modules['filter_shape[cone]'] = self.filter_cone
        self.modules['filter_size[large]'] = self.filter_large
        self.modules['filter_size[small]'] = self.filter_small
        self.modules['filter_size[medium]'] = self.filter_medium
        #self.modules['greater_than'] = self.greater_than
        #self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        self.modules['query_color'] = self.query_color
        self.modules['query_material'] = self.query_material
        self.modules['query_shape'] = self.query_shape
        self.modules['query_size'] = self.query_size
        self.modules['relate[behind]'] = self.relate_behind
        self.modules['relate[front]'] = self.relate_front
        self.modules['relate[left]'] = self.relate_left
        self.modules['relate[right]'] = self.relate_right
        self.modules['same_color'] = self.same_color
        self.modules['same_material'] = self.same_material
        self.modules['same_shape'] = self.same_shape
        self.modules['same_size'] = self.same_size
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique
        
    
    def filter_blue(self, scene,  constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'blue':
                    output.append(o)
            regions_blue = self.getRegions_feature(constraint, 'color', 'blue')
            region_ans = self.intersection(region_prev, regions_blue)
            return output, region_ans
        return 'error', 'error'
    
    def filter_brown(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'brown':
                    output.append(o)
            regions_brown = self.getRegions_feature(constraint, 'color', 'brown')
            region_ans = self.intersection(region_prev, regions_brown)
            return output, region_ans    
            
        return 'error', 'error'
    
    def filter_cyan(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'cyan':
                    output.append(o)
            regions_cyan = self.getRegions_feature(constraint, 'color', 'cyan')
            region_ans = self.intersection(region_prev, regions_cyan)
            return output, region_ans    
        return 'error', 'error'
    
    def filter_gray(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'gray':
                    output.append(o)
            regions_gray = self.getRegions_feature(constraint, 'color', 'gray')
            region_ans = self.intersection(region_prev, regions_gray)
            return output, region_ans
        return 'error', 'error'
    
    def filter_green(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'green':
                    output.append(o)
            regions_green = self.getRegions_feature(constraint, 'color', 'green')
            region_ans = self.intersection(region_prev, regions_green)
            return output, region_ans
        return 'error', 'error'
    
    def filter_purple(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'purple':
                    output.append(o)
            regions_purple = self.getRegions_feature(constraint, 'color', 'purple')
            region_ans = self.intersection(region_prev, regions_purple)
            return output, region_ans
        return 'error', 'error'
    
    def filter_red(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'red':
                    output.append(o)
            regions_red = self.getRegions_feature(constraint, 'color', 'red')
            region_ans = self.intersection(region_prev, regions_red)
            return output, region_ans
        return 'error', 'error'
    
    def filter_yellow(self, scene, constraint, region_prev, *_):
        
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'yellow':
                    output.append(o)
            regions_yellow = self.getRegions_feature(constraint, 'color', 'yellow')
            region_ans = self.intersection(region_prev, regions_yellow)
            return output, region_ans
        return 'error', 'error'

    def filter_coral(self, scene, constraint, region_prev, *_):
        
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'coral':
                    output.append(o)
            regions_coral = self.getRegions_feature(constraint, 'color', 'coral')
            region_ans = self.intersection(region_prev, regions_coral)
            return output, region_ans
        return 'error', 'error'
    
    def filter_rubber(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'rubber':
                    output.append(o)
            regions_rubber = self.getRegions_feature(constraint, 'material', 'rubber')
            region_ans = self.intersection(region_prev, regions_rubber)
            return output, region_ans
        return 'error', 'error'
    
    def filter_metal(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'metal':
                    output.append(o)
            regions_metal = self.getRegions_feature(constraint, 'material', 'metal')
            region_ans = self.intersection(region_prev, regions_metal)
            return output, region_ans
        return 'error', 'error'
    
    def filter_cube(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cube':
                    output.append(o)
            regions_cube = self.getRegions_feature(constraint, 'shape', 'cube')
            region_ans = self.intersection(region_prev, regions_cube)
            return output, region_ans
        return 'error', 'error'
    
    def filter_cylinder(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cylinder':
                    output.append(o)
            regions_cylinder = self.getRegions_feature(constraint, 'shape', 'cylinder')
            region_ans = self.intersection(region_prev, regions_cylinder)
            return output, region_ans
        return 'error', 'error'
    
    def filter_sphere(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'sphere':
                    output.append(o)
            regions_sphere = self.getRegions_feature(constraint, 'shape', 'sphere')
            region_ans = self.intersection(region_prev, regions_sphere)
            return output, region_ans
        return 'error', 'error'

    def filter_cone(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cone':
                    output.append(o)
            regions_cone = self.getRegions_feature(constraint, 'shape', 'cone')
            region_ans = self.intersection(region_prev, regions_cone)
            return output, region_ans
        return 'error', 'error'


    def filter_large(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'large':
                    output.append(o)
            regions_large = self.getRegions_feature(constraint, 'size', 'large')
            region_ans = self.intersection(region_prev, regions_large)
            return output, region_ans
        return 'error', 'error'
    
    def filter_small(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'small':
                    output.append(o)
            regions_small = self.getRegions_feature(constraint, 'size', 'small')
            region_ans = self.intersection(region_prev, regions_small)
            return output, region_ans
        return 'error', 'error'

    def filter_medium(self, scene, constraint, region_prev, *_):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'medium':
                    output.append(o)
            regions_med = self.getRegions_feature(constraint, 'size', 'medium')
            region_ans = self.intersection(region_prev, regions_med)
            return output, region_ans
        return 'error', 'error'
    
#    def greater_than(self, integer1, integer2):
#        if type(integer1) == int and type(integer2) == int:
#            if integer1 > integer2:
#                return 'yes'
#            else:
#                return 'no'
#        return 'error'
#    
#    def less_than(self, integer1, integer2):
#        if type(integer1) == int and type(integer2) == int:
#            if integer1 < integer2:
#                return 'yes'
#            else:
#                return 'no'
#        return 'error'
    
    def intersect(self, scene1, constraint, region_prev, scene2, region_temp):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            region_ans = self.intersection(region_prev, region_temp)
            return output, region_ans
        return 'error', 'error'
    
    def query_color(self, obj, constraint, region_prev,*_):
        if type(obj) == dict and 'color' in obj:
            return obj['color'], region_prev
        elif type(obj) == dict and len(obj) == 0:
            return -1, region_prev
        return 'error', 'error'
    
    def query_material(self, obj, constraint, region_prev,*_):
        if type(obj) == dict and 'material' in obj:
            return obj['material'], region_prev
        elif type(obj) == dict and len(obj) == 0:
            return -1, region_prev
        return 'error', 'error'
    
    def query_shape(self, obj, constraint, region_prev, *_):
        if type(obj) == dict and 'shape' in obj:
            return obj['shape'], region_prev
        elif type(obj) == dict and len(obj) == 0:
            return -1, region_prev
        return 'error', 'error'
    
    def query_size(self, obj, constraint, region_prev, *_):
        if type(obj) == dict and 'size' in obj:
            return obj['size'], region_prev
        elif type(obj) == dict and len(obj) == 0:
            return -1, region_prev
        return 'error', 'error'
    
    def relate_behind(self, obj,  constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] > obj['position'][1]:
                    output.append(o)
            regions_behind = self.findRegionsRel(region_prev, 'behind')
            region_ans = self.intersection(regions_behind, region_temp)
            return output, region_ans
        elif type(obj) == dict and len(obj)==0 and type(scene) == list:
            output = []
            regions_behind = self.findRegionsRel(region_prev, 'behind')
            region_ans = self.intersection(regions_behind, region_temp)
            return output, region_ans
        return 'error', 'error'
    
    def relate_front(self, obj,  constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] < obj['position'][1]:
                    output.append(o)
            regions_front = self.findRegionsRel(region_prev, 'front')
            region_ans = self.intersection(regions_front, region_temp)
            return output, region_ans
        elif type(obj) == dict and len(obj)==0 and type(scene) == list:
            output = []
            regions_front = self.findRegionsRel(region_prev, 'front')
            region_ans = self.intersection(regions_front, region_temp)
            return output, region_ans
        return 'error', 'error'
    
    def relate_left(self, obj,  constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] < obj['position'][0]:
                    output.append(o)
            regions_left = self.findRegionsRel(region_prev, 'left')
            region_ans = self.intersection(regions_left, region_temp)
            return output, region_ans
            
        elif type(obj) == dict and len(obj) == 0 and type(scene) == list:
            output = []
            regions_left = self.findRegionsRel(region_prev, 'left')
            region_ans = self.intersection(regions_left, region_temp)
            return output, region_ans
        return 'error', 'error'
    
    def relate_right(self, obj,  constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] > obj['position'][0]:
                    output.append(o)
            regions_right = self.findRegionsRel(region_prev, 'right')
            region_ans = self.intersection(regions_right, region_temp)
            return output, region_ans
        elif type(obj) == dict and len(obj) == 0 and type(scene) == list:
            output = []
            regions_right = self.findRegionsRel(region_prev, 'right')
            region_ans = self.intersection(regions_right, region_temp)
            return output, region_ans
        return 'error', 'error'
    
    def same_color(self, obj, constraint, region_prev, scene,  region_temp):
        if type(obj) == dict and 'color' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == obj['color'] and o['id'] != obj['id']:
                    output.append(o)
            regions_color = self.getRegions_feature(constraint, 'color', obj['color'])
            region_ans = self.intersection(regions_color, region_temp)
            return output, region_ans
        elif type(obj) == dict and len(obj)==0 and type(scene) == list:
            output = []
            regions_ans = []
            poss_col = []
            for r in region_prev:
                P_cols = self.getFeatureVals_Region(r, constraint, 'color')
                for s in P_cols:
                    if s not in poss_col:
                        poss_col.append(s)
            #print("possible shapes at 1,2 ::", poss_shapes)
            for m in poss_col:
                regions = self.getRegions_feature(constraint, 'color', m)
                for reg in regions:
                    if reg not in regions_ans:
                        regions_ans.append(reg)
            return output, regions_ans
        return 'error', 'error'
    
    def same_material(self, obj,  constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'material' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == obj['material'] and o['id'] != obj['id']:
                    output.append(o)
            regions_material = self.getRegions_feature(constraint, 'material', obj['material'])
            region_ans = self.intersection(regions_material, region_temp)
            return output, region_ans
        elif type(obj) == dict and len(obj)==0 and type(scene) == list:
            output = []
            regions_ans = []
            poss_mat = []
            for r in region_prev:
                P_mats = self.getFeatureVals_Region(r, constraint, 'material')
                for s in P_mats:
                    if s not in poss_mat:
                        poss_mat.append(s)
            #print("possible shapes at 1,2 ::", poss_shapes)
            for m in poss_mat:
                regions = self.getRegions_feature(constraint, 'material', m)
                for reg in regions:
                    if reg not in regions_ans:
                        regions_ans.append(reg)
            return output, regions_ans
        return 'error', 'error'
    
    def same_shape(self, obj, constraint, region_prev, scene, region_temp):
        #print("In same shape::", obj)
        #print("Region_prev:", region_prev)
        #print("Region_temp:", region_temp)
        if type(obj) == dict and 'shape' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == obj['shape'] and o['id'] != obj['id']:
                    output.append(o)
            regions_shape = self.getRegions_feature(constraint, 'shape', obj['shape'])
            
            region_ans = self.intersection(regions_shape, region_temp)
            #print("Shape we are looking for:", obj['shape'], obj)
            return output, region_ans
            
        elif type(obj) == dict and len(obj)==0 and type(scene) == list:
            output = []
            regions_ans = []
            poss_shapes = []
            for r in region_prev:
                P_shape = self.getFeatureVals_Region(r, constraint, 'shape')
                for s in P_shape:
                    if s not in poss_shapes:
                        poss_shapes.append(s)
            #print("possible shapes at 1,2 ::", poss_shapes)
            for shape in poss_shapes:
                regions = self.getRegions_feature(constraint, 'shape', shape)
                for reg in regions:
                    if reg not in regions_ans:
                        regions_ans.append(reg)
            ##region_ans = self.intersection(region_prev, region_temp)
            return output, regions_ans
        return 'error', 'error'
    
    def same_size(self, obj, constraint, region_prev, scene, region_temp):
        if type(obj) == dict and 'size' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == obj['size'] and o['id'] != obj['id']:
                    output.append(o)
            regions_size = self.getRegions_feature(constraint, 'size', obj['size'])
            region_ans = self.intersection(regions_size, region_temp)
            return output, region_ans        
        elif type(obj) == dict and len(obj) == 0 and type(scene) == list:
            output = []
            regions_ans = []
            poss_sizes = []
            for r in region_prev:
                P_sizes = self.getFeatureVals_Region(r, constraint, 'size')
                for s in P_sizes:
                    if s not in poss_sizes:
                        poss_sizes.append(s)
            #print("possible shapes at 1,2 ::", poss_shapes)
            for size in poss_sizes:
                regions = self.getRegions_feature(constraint, 'size', size)
                for reg in regions:
                    if reg not in regions_ans:
                        regions_ans.append(reg)
            return output, regions_ans
        return 'error', 'error'
    
    def union(self, scene1, constraint, region_prev, scene2, region_temp):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            region_ans = list(region_prev)
            for r in region_temp:
                if r not in region_prev:
                    region_ans.append(r)
            return output, region_ans
        return 'error', 'error'
    
    def unique(self, scene,constraint, region_prev, *_):
        if type(scene) == list and len(scene) > 0:
            return scene[0], region_prev
        if type(scene) == list and len(scene) == 0:
            empty_dict  = {}
            return empty_dict, region_prev
        return 'error', 'error'
