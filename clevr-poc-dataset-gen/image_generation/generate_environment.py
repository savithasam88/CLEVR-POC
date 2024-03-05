#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import random
import copy, os
import sys, math

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


PROPERTIES = ['shape', 'color', 'material', 'size']
domain = {}
domain['color'] = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'] 
domain['material'] = ['rubber', 'metal']
domain['shape'] = ['cube', 'cylinder', 'sphere', 'cone']
domain['size'] = ['large', 'small', 'medium']

region = [0,1,2,3] #4,5,6,7,8]


class Template:
    def __init__(self, form, var, val_var):
        self.form = form
        self.var = var
        self.val_var = val_var
    def instantiate(self, *region):
        cal_var = []
       
        for var in self.val_var:
            func_arg = copy.deepcopy(self.val_var[var])
            if var!=0:
                for a in range(1, len(func_arg)):
                    if type(func_arg[a]) == int:
                        func_arg[a] = cal_var[func_arg[a]]
            else:
                for a in range(1, len(func_arg)):
                    if func_arg[a]=='region' and region:
                        func_arg[a] = region[0]
                   
            val = func_arg[0](*func_arg[1:])
            cal_var.append(val)
        form = self.form
        for v in range(len(self.var)):
            form = form.replace(self.var[v], str(cal_var[v]))
        return form

def generateConstraints(templates, negation, across, within):
    constraints = ""
    #Generate one (within) region constraint for each of the 9 regions
    
    
    for r in region:
        #wn = within+negation
        region_cons = random.choice(within)
        c = templates[region_cons].instantiate(r)
        c = c.strip()
        
        c_split =  list(filter(None, c.split('.'))) #c.split('.')
        
        for con in range(len(c_split)):
            #input(c_split[con])
            #if c_split[con]!='':
            constraints = constraints + c_split[con] + "." + "\n"
            
        #mandatory negation constraints
        
        t = random.choice([1, 2])
        for n  in range(0,t):
            cons_num = random.choice(negation)
            c = templates[cons_num].instantiate(r)
            c_split = list(filter(None, c.split('.')))
            for con in range(len(c_split)):
                #if c_split[con]!='':
                constraints = constraints + c_split[con] + "." + "\n"
                
    #Generate 3 across region constraints
    t = random.choice([1, 2, 3])
    for i in range(t):
        n = random.choice(across)
        c = templates[n].instantiate()
        c_split = list(filter(None, c.split('.')))
        
        for con in range(len(c_split)):
            #if c_split[con]!='':
            constraints = constraints + c_split[con] + "." + "\n"
            
    return constraints
       

def createTemplateInstance(templates_list):

    #all objects in R' should have value V1' for P1' and value V2' for P2'
    template_1 = templates_list[0]
    #":- object(X), at(X, R'), not hasProperty(X, P1', V1'). :- object(X), at(X, R'), not hasProperty(X, P2', V2')."
    vars1 = ["R'", "P1'", "V1'", "P2'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 1, PROPERTIES]
    val_var[4] = [lambda x: random.choice(domain[x]), 3]
    
    t1 = Template(template_1, vars1, val_var)
    #t1_instance = t1.instantiate()
    #print(t1_instance)
    
    #Negation: all objects in R' should not have value V1' for P1' and value V2' for P2'
    template_2 = templates_list[1]
    #":- object(X), at(X, R'), hasProperty(X, P1', V1'). :- object(X), at(X, R'), hasProperty(X, P2', V2')."
    vars1 = ["R'", "P1'", "V1'", "P2'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 1, PROPERTIES]
    val_var[4] = [lambda x: random.choice(domain[x]), 3]
    
    t2 = Template(template_2, vars1, val_var)
    
    #Negation: all objects in R' should not have value V1', V2' for P1' 
    template_3 = templates_list[2]
    #":- object(X), at(X, R'), hasProperty(X, P1', V1'). :- object(X), at(X, R'), hasProperty(X, P1', V2')."
    vars1 = ["R'", "P1'", "V1'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y,z : random.choice(list(filter(lambda x: (x != y), domain[z]))), 2, 1]
    
    t3 = Template(template_3, vars1, val_var)
    #t3_instance = t3.instantiate()
    #print(t3_instance)
    
    #all objects in R' should have value V1' for P1' or value V2' for P2'
    template_4 = templates_list[3]
    #":- object(X), at(X, R'), not hasProperty(X, P1', V1'), not hasProperty(X, P2', V2')."
    vars1 = ["R'", "P1'", "V1'", "P2'", "V2'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 1, PROPERTIES]
    val_var[4] = [lambda x: random.choice(domain[x]), 3]
    t4 = Template(template_4, vars1, val_var)
    #t2_instance = t2.instantiate()
    #print(t2_instance)
    
    #all objects in R' should not have value V1' for P1'
    template_5 = templates_list[4]
    #":- object(X), at(X, R'), hasProperty(X, P1', V1')."
    vars1 = ["R'", "P1'", "V1'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    t5 = Template(template_5, vars1, val_var)
    #t3_instance = t3.instantiate()
    #print(t3_instance)
    
    #there are exactly N' objects with P1' = V1' in R1', n=2.
    template_6 = templates_list[5]
    #":- #count{hasProperty(X, P1', V1') : object(X), at(X, R1')}!=N'."
    vars1 = ["R1'", "P1'", "V1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : region, 'region']
    val_var[1] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[2] = [lambda x : random.choice(domain[x]), 1]
    val_var[3] = [lambda x: random.choice(x), [1,2]]
    t6 = Template(template_6, vars1, val_var)
    #t8_instance = t8.instantiate()
    #print(t8_instance)
    
    
    
    #|{set of objects with P1'=V in R1'} intersetcion {set of objects with P1' = V in R2'}| >= N'
    template_7 = templates_list[6]
    #":- #count{sameProperty(X1, X2, P1'): object(X1), object(X2), at(X1, R1'), at(X2, R2')}<N'."
    vars1 = ["R1'", "R2'",  "P1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda x: random.choice(x), [2,4]]
    t7 = Template(template_7, vars1, val_var)
    #t4_instance = t4.instantiate()
    #print(t4_instance)
    
    #|{set of objects with P2'=V2' and P1'=V in R1'} intersetcion {set of objects with P2' = V2', P1' = V in R2'}| >= N'
    template_8 = templates_list[7]
    #":- #count{sameProperty(X1, X2, P1'): object(X1), object(X2), at(X1, R1'), at(X2, R2'), hasProperty(X1, P2', V2'), hasProperty(X2, P2', V2')}<N'."
    vars1 = ["R1'", "R2'",  "P1'", "P2'", "V2'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 2, PROPERTIES]
    val_var[4] = [lambda x : random.choice(domain[x]), 3]
    val_var[5] = [lambda x: random.choice(x), [2,4]]
    t8 = Template(template_8, vars1, val_var)
    #t8_instance = t8.instantiate()
    #print(t5_instance)
    
    
    #|{set of objects with P1'=V in R1'} intersetcion {set of objects with P1' = V in R2'}| <N'
    template_9 = templates_list[8]
    #":- #count{sameProperty(X1, X2, P1'): object(X1), object(X2), at(X1, R1'), at(X2, R2')}>=N'."
    vars1 = ["R1'", "R2'",  "P1'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda x: random.choice(x), [2, 4]]
    t9 = Template(template_9, vars1, val_var)
    #t9_instance1 = t9.instantiate()
    #t6_instance2 = t6.instantiate()
    #print(t6_instance1)
    
    #|{set of objects with P2'=V2' and P1'=V in R1'} intersetcion {set of objects with P2' = V2', P1' = V in R2'}| <N'
    template_10 = templates_list[9]
    #":- #count{sameProperty(X1, X2, P1'): object(X1), object(X2), at(X1, R1'), at(X2, R2'), hasProperty(X1, P2', V2'), hasProperty(X2, P2', V2')}>=N'."
    vars1 = ["R1'", "R2'",  "P1'", "P2'", "V2'", "N'"]
    val_var = {}
    val_var[0] = [lambda region : random.choice(region), region]
    val_var[1] = [lambda y, region: random.choice(list(filter(lambda x: (x != y), region))), 0, region]
    val_var[2] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[3] = [lambda y, PROPERTIES: random.choice(list(filter(lambda x: (x != y), PROPERTIES))), 2, PROPERTIES]
    val_var[4] = [lambda x : random.choice(domain[x]), 3]
    val_var[5] = [lambda x: random.choice(x), [2, 4]]
    t10 = Template(template_10, vars1, val_var)
    #t7_instance = t7.instantiate()
    #print(t7_instance)
    
    #Generic constraint - an object should not have value V1' for P1'
    template_11 = templates_list[10]
    #":- object(X), hasProperty(X, P1', V1')."
    vars1 = ["P1'", "V1'"]
    val_var = {}
    val_var[0] = [lambda PROPERTIES : random.choice(PROPERTIES), PROPERTIES]
    val_var[1] = [lambda x : random.choice(domain[x]), 1]
    t11 = Template(template_11, vars1, val_var)

    templates = [t1, t2, t3, t8, t4, t5, t6, t7, t8, t9, t10, t11]
    negation = [1,2,4]
    across = [6,7,8,9,10]
    within = [0,3,5]
    return templates, negation, across, within

def generateEnvironment(args, environment_constraints_dir, num_objects, env_id):
    num_objects = num_objects - 1
    templates_list=[]
    file1 = open(args.constraint_template_path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()       
        templates_list.append(line)
    templates, negation, across, within  = createTemplateInstance(templates_list)
    file1.close()
    file1 = open(args.general_constraints_path, 'r')
    Lines = file1.readlines()
    background = ""
    for line in Lines:
        background = background+line
    file1.close()
    background = background+"\n"+"object(0.."+str(num_objects)+")."+"\n"    
    satisfiable = False
    while(not(satisfiable)):
        asp_file = open(os.path.join(environment_constraints_dir, str(env_id)+".lp"), "w")
        constraints = generateConstraints(templates, negation, across, within)
        asp_code = background+constraints+"\n"+"#show hasProperty/3. #show at/2."
        n1 = asp_file.write(asp_code)
        asp_file.close()
        asp_command = 'clingo 1'  + ' ' + os.path.join(environment_constraints_dir, str(env_id)+".lp")
        output_stream = os.popen(asp_command)
        output = output_stream.read()
        output_stream.close()
        answers = output.split('Answer:')
        #print("Answers:", answers)
        answers = answers[1:]
        count = 0
        for answer in answers:
            count = count+1
            if(count>=1):
                satisfiable = True
                print("Satisfiable")
                break
    
#----------------------------------------------------------------------------------------------------------
def getObjects(preds, obj_rm, given_query):
    #object(color, material, shape, size, region)
    complete = {}
    incomplete = {}
    for pred in preds:
        if 'hasProperty' in pred:
            pred_split = pred.split("(")
            obj_prop_val = pred_split[1].split(",")
            obj  = obj_prop_val[0]
            prop = obj_prop_val[1]
            val = obj_prop_val[2][0:len(obj_prop_val[2])-1]
            try:
                complete[int(obj)][prop] = val
            except:
                complete[int(obj)] = {}
                complete[int(obj)][prop] = val
            if int(obj)!=obj_rm:
                try:
                    incomplete[int(obj)][prop] = val
                except:
                    incomplete[int(obj)] = {}
                    incomplete[int(obj)][prop] = val
            else:
                if prop in given_query:
                    try:
                        incomplete[int(obj)][prop] = val
                    except:
                        incomplete[int(obj)] = {}
                        incomplete[int(obj)][prop] = val
                    
        elif 'at(' in pred:
            pred_split = pred.split("(")
            obj_reg = pred_split[1].split(",")
            obj = obj_reg[0]
            reg = obj_reg[1][0:len(obj_reg[1])-1]
            try:
                complete[int(obj)]['region'] = reg
            except:
                complete[int(obj)] = {}
                complete[int(obj)]['region'] = reg
            if int(obj)!=obj_rm:
                try:
                    incomplete[int(obj)]['region'] = reg
                except:
                    incomplete[int(obj)] = {}
                    incomplete[int(obj)]['region'] = reg
    return complete, incomplete


#---------------------------------------------------------------------------------------------------------    
def getQA(query_attribute, given_query, complete, incomplete_details, obj_rm, environment_constraints_dir):
     for g in given_query:
         for pred in incomplete_details:
             if g in pred:
                 complete = complete+"\n"+pred+"."
            
     complete_qa = complete+"\n missing(V):-hasProperty("+str(obj_rm)+","+query_attribute+",V)."     
         
     complete_qa = complete_qa+"#show missing/1."
     temp_file = environment_constraints_dir + "/incomplete_"+query_attribute+".lp"
     file1 = open(temp_file, 'w')
     n1 = file1.write(complete_qa)
     file1.close()
     asp_command = 'clingo 0'  + ' ' + temp_file
     output_stream = os.popen(asp_command)
     output = output_stream.read()
     output_stream.close()
     answers = output.split('Answer:')
     answers = answers[1:]
     possible_values = []
     for answer_index, answer in enumerate(answers):
         ans = answer.split('\n')[1].split(' ')
         val = ans[0][8:(len(ans[0])-1)]
         if (val not in possible_values):
             possible_values.append(val)
     temp_path = os.path.join(temp_file)
     if os.path.isfile(temp_path):
         os.remove(temp_path)
     if len(possible_values) == len(domain[query_attribute]):
         return None
     elif len(possible_values) < len(domain[query_attribute]):
         return possible_values 

    
#----------------------------------------------------------------------------------------------
def balance_queryAttribute_numImages(num_images_per_qa, max_number_of_images_per_qa):
  for i in range(len(num_images_per_qa)):
     if num_images_per_qa[i] < max_number_of_images_per_qa[i]:
      return i
  return random.randint(0, 3) 
    
#---------------------------------------------------------------------------------------------
def chooseGiven(props, query_attribute, n1):
    #Choose n1 props that is not query_attribute
    given = []
    allowed = copy.deepcopy(props)
    allowed.remove(query_attribute)
    given_comb = [[0], [0, 1],[1], [0, 2], [2], [1, 2], [0,1,2]] 
    chosen_comb = given_comb[n1]
    for k in chosen_comb:
    	given.append(allowed[k]) 
    return given     

#----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#Creates an incomplete scene graph by removing obj_rm and decides on the query_attribute and given_attributes 
#based on possible_sols for the query () 

def createQuery_Incomplete(asp_file, preds, obj_rm, environment_constraints_dir,args, num_image_per_qa, max_number_of_images_per_qa):
     props = ['color', 'shape', 'size', 'material']
     #max_number_of_images_per_qa = math.ceil(args.num_images/4.0)
     file2 = open(asp_file, 'r')
     Lines = file2.readlines()
     complete = ""
     for line in Lines:
         if "#show" in line:
             continue
         complete = complete+line
     file2.close()
     complete = complete+ '\n'+ ':-not missing(_).'+'\n'
     #Add details about current scene graph (except about obj_rm) in preds
     incomplete_details = []
     
     for pred in preds:
     
         if pred.split("(")[1][0] == str(obj_rm):
            incomplete_details.append(pred)
            continue
         complete = complete+"\n"+pred+"."
     query_attr = ""
     possible_sols = []
     given_query = []
     for k in range(10):
         query_attribute = random.choice(props) #balance_queryAttribute_numImages(num_image_per_qa, max_number_of_images_per_qa) 
         n1 = random.randint(0, 2)
         given = chooseGiven(props, query_attribute, n1)
         
         possible_sols_qa = getQA(query_attribute, given, complete, incomplete_details, obj_rm, environment_constraints_dir)
         if possible_sols_qa!=None :
             #question, structured = generate_question(args,query_attribute, given, obj_rm, preds, possible_sols_qa)
             #print("Question generated::")
             return query_attribute, possible_sols_qa, given
             
                 
     return query_attr, possible_sols, given_query
        
         
#-------------------------------------------------------------------------------------------------------------
#1. Creates a scene graph that conforms to the environment - constraint_type_index - by generating answer sets 
# to the program that takes number of objects and the constraints as input  - a set of 1000000 answers are considered , each a scene graph
#env_answers maps the environment to these answer sets.
#2. Creates an incomplete scene graph with an obj_interest, query_attribute, given_attributes and set of possible soluions for it.
      
        
def getSceneGraph_data(num_objects, constraint_type_index, env_answers, environment_constraints_dir, args, start_from, num_image_per_qa, max_number_of_images_per_qa):
    props = ['color', 'shape', 'size', 'material']
    answers = env_answers[constraint_type_index]
        
    query_attr = "" 
    given_query = []

    objects =list(range(num_objects))
    answer_index = start_from[constraint_type_index]
    answer = answers[answer_index]
    
    preds = answer.split('\n')[1].split(' ')
    obj_rm = random.choice(objects)
    query_attr_index = balance_queryAttribute_numImages(num_image_per_qa, max_number_of_images_per_qa)
    query_attr = props[query_attr_index]
    given_query = chooseGiven(props, query_attr, 0)
    start_from[constraint_type_index] = answer_index+1 
    complete, incomplete = getObjects(preds, obj_rm, given_query)
    return complete, incomplete, query_attr, given_query, obj_rm, start_from
                
    
def getSceneGraph_constraint(num_objects, constraint_type_index, env_answers, environment_constraints_dir, args):
    MAX_NUMBER_OF_ANSWERS = 1000000
    asp_file = environment_constraints_dir + str(constraint_type_index)+".lp"
    ASP_FILE_PATH = os.path.join(asp_file)
    asp_command = 'clingo ' + str(MAX_NUMBER_OF_ANSWERS) + ' ' + ASP_FILE_PATH
    output_stream = os.popen(asp_command)
    output = output_stream.read()
    output_stream.close()
    ## parsing answer sets

    answers = output.split('Answer:')
    answers = answers[1:]
    random.shuffle(answers)

    if len(answers) <=  int(math.ceil(10.0/100*(MAX_NUMBER_OF_ANSWERS))):
        number_sample = len(answers)
    else:
        number_sample = int(math.ceil(10.0/100*(MAX_NUMBER_OF_ANSWERS)))
        
    answers = random.sample(answers, number_sample)
    return answers
