import math, sys, random, argparse, json, os, tempfile
import collections 
import copy
import gc
import pickle

from pathlib import Path
path_root = Path(__file__).parents[1]
path_current = Path(__file__).parents[0]
sys.path.append(str(path_root))
sys.path.append(str(path_current))



from image_generation import scene_info, blender
from generate_dataset import parser
from generate_environment import generateEnvironment, getSceneGraph_data, getSceneGraph_constraint
from question_generation.generate_questions import generate_question


VERY_LARG_NUMBER = 1000000

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)



def directory_management(main_dir):
  image_dir = os.path.join(main_dir, args.image_dir, args.split)
  scene_dir = os.path.join(main_dir, args.scene_dir, args.split)

  
  if not os.path.isdir(image_dir):
    os.makedirs(image_dir) 

  if not os.path.isdir(scene_dir):
    os.makedirs(scene_dir) 

   

  num_digits = 6

  prefix = '%s_' % (args.filename_prefix)
  image_temp = '%s%%0%dd.png' % (prefix, num_digits)
  scene_temp = '%s%%0%dd.json' % (prefix, num_digits)
  
  scene_template = os.path.join(scene_dir, scene_temp)
  img_template = os.path.join(image_dir, image_temp)



  return scene_dir, img_template, scene_template


##---------------------------------------------------------------------------------------------------------------------------
def main(args):
  

  complete_scene_dir, complete_img_template, complete_scene_template = directory_management(args.complete_data_dir)
  incomplete_scene_dir, incomplete_img_template, incomplete_scene_template = directory_management(args.incomplete_data_dir)

  question_dir = os.path.join(args.incomplete_data_dir, args.question_dir, args.split)
  if not os.path.isdir(question_dir):
    os.makedirs(question_dir) 

  num_digits = 6
  prefix = '%s_' % (args.filename_prefix)
  question_temp = '%s%%0%dd.json' % (prefix, num_digits)
  question_template = os.path.join(question_dir, question_temp)




  environment_constraints_dir = os.path.join(args.environment_constraints_dir)
  if not os.path.isdir(environment_constraints_dir):
    os.makedirs(environment_constraints_dir)     


  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
  
  

 

  if args.phase_constraint == 1:
    num_images = args.num_constraint_types
    possible_num_objects = [i for i in range(args.min_objects, args.max_objects+1)]

    objNum_env = {i:[] for i in range(args.min_objects,args.max_objects+1)}
    env_answers = {}
    updated = {i:0 for i in range(args.num_constraint_types)}
    num_env_per_numObj = [0 for i in range(args.min_objects, args.max_objects+1)] 
    max_number_of_env_per_numObj = args.num_constraint_types/len(num_env_per_numObj)
    env_id = 0
    
  else:
    print('Loading environments...')
    num_images = args.num_images
    #Load env details - give path!!
    env_ans_file = open(os.path.join(environment_constraints_dir,"env_answers.obj"),"rb")
    env_answers = pickle.load(env_ans_file)
    env_ans_file.close()
    #for e in env_answers:
    #	print('Env e:', e, len(env_answers[e]))



    
    objNum_env_file = open(os.path.join(environment_constraints_dir,"objNum_env.obj"),"rb")
    objNum_env = pickle.load(objNum_env_file)
    objNum_env_file.close()
    max_number_of_images_per_constraint = math.ceil(num_images/args.num_constraint_types)
    max_number_of_images_per_qa = []
    max_number_of_images_per_qa.append(math.ceil((40/100)*num_images)) #color
    max_number_of_images_per_qa.append(math.ceil((40/100)*num_images)) #shape
    max_number_of_images_per_qa.append(math.ceil((20/100)*num_images)) #size
    max_number_of_images_per_qa.append(math.ceil((20/100)*num_images)) #material
    
    
    

    if args.start_idx == 0:
      possible_num_objects = [i for i in range(args.min_objects, args.max_objects+1)]
      num_image_per_constraint_type = [0 for ind in range(args.num_constraint_types)]
      num_image_per_qa = [0 for ind in range(4)]
      if args.split == 'training':
          updated = {i:0 for i in range(args.num_constraint_types)}
          
      else:
          updated_file = open(os.path.join(environment_constraints_dir,"updated.obj"),"rb")
          updated = pickle.load(updated_file)
          updated_file.close()
    else:
      updated_file = open(os.path.join(environment_constraints_dir,"updated.obj"),"rb")
      updated = pickle.load(updated_file)
      updated_file.close()
      
      num_image_per_constraint_type_file = open(os.path.join(environment_constraints_dir,"num_image_per_constraint_type.pickle"),'rb')
      num_image_per_constraint_type = pickle.load(num_image_per_constraint_type_file)
      num_image_per_constraint_type_file.close()
      
      num_image_per_qa_file = open(os.path.join(environment_constraints_dir,"num_image_per_qa.pickle"),'rb')
      num_image_per_qa = pickle.load(num_image_per_qa_file)
      num_image_per_qa_file.close()
      
      possible_num_objects_file = open(os.path.join(environment_constraints_dir,"possible_num_objects.pickle"),'rb')
      possible_num_objects = pickle.load(possible_num_objects_file)
      possible_num_objects_file.close()


      #Loading question templates
    templates = {}
    num_loaded_templates = 0
    for fn in os.listdir(os.path.join(str(path_root), "question_generation", args.template_dir)):
        if not fn.endswith('.json'): continue
        with open(os.path.join(str(path_root), "question_generation", args.template_dir, fn), 'r') as f:
          for i, template in enumerate(json.load(f)):
              num_loaded_templates = num_loaded_templates + 1
              key = (fn, i)
              templates[key] = template
          print('Read %d templates from disk' % num_loaded_templates)
  
    num_questions_per_template_type = {}
    for key in templates:
        num_questions_per_template_type[key] = 0

    max_number_of_questions_per_template = math.ceil(args.num_images/args.num_templates) 
      

  
  i = args.start_idx
  args.render_batch_size = args.num_images


  while i < args.num_images:
        complete_scene_graph = {} 
        incomplete_scene_graph = {} 
        query_attribute = "" 
        given_query = [] 
        complete_scene = None
        end_of_process = False
        
        #Data Generation Phase
        if args.phase_constraint == 1:
                    
            index_num_obj = balance_env_numObj(num_env_per_numObj, max_number_of_env_per_numObj)
            num_objects = possible_num_objects[index_num_obj]
            generateEnvironment(args, environment_constraints_dir, num_objects, env_id)
            
            constraint_type_index = env_id
            updated_answers = getSceneGraph_constraint(num_objects, constraint_type_index, env_answers, environment_constraints_dir, args)
            env_answers[constraint_type_index] = updated_answers
            objNum_env[num_objects].append(env_id)
            num_env_per_numObj[index_num_obj] = num_env_per_numObj[index_num_obj]+1
            env_id = env_id +1
            #i = i+1
        
        else:
            
            no_question = True
            while(no_question):
                #input(no_question)
                complete_scene_image_path = complete_img_template % i
                incomplete_scene_image_path = incomplete_img_template % i
            

                complete_scene_path = complete_scene_template % i
                incomplete_scene_path = incomplete_scene_template % i
                
                question_path = question_template % i
                
                #Choose an environment type
                if len(possible_num_objects)== 0:
                    num_images_left = args.num_images - i
                    max_number_of_images_per_constraint = max_number_of_images_per_constraint  + math.ceil(num_images_left/args.num_constraint_types) 
                    possible_num_objects = [num for num in range(args.min_objects, args.max_objects+1)]
                  
                num_objects = random.choice(possible_num_objects)
                list_env = objNum_env[num_objects]
                #input(num_objects)
                constraint_type_index = balance_constraint_type(list_env, num_image_per_constraint_type, max_number_of_images_per_constraint)
                
                if constraint_type_index == None:
                  possible_num_objects.remove(num_objects)
                  continue
              
                for e in updated:
                    if updated[e] == len(env_answers[e])-1:
                        num_image_per_constraint_type[e] = VERY_LARG_NUMBER
                        
                
                temp = [ind for ind in range(len(num_image_per_constraint_type)) if num_image_per_constraint_type[ind] == VERY_LARG_NUMBER] 
                if len(temp) == len(num_image_per_constraint_type):
                	end_of_process = True
                	break
                """     		       
                if (num_image_per_constraint_type.count(num_image_per_constraint_type[0]) == len(num_image_per_constraint_type)):
                    if(num_image_per_constraint_type[0] == VERY_LARG_NUMBER):
                        end_of_process = True
                        break
                """
            
                #Generate a scene from that environment
                print(updated)
                print(constraint_type_index)
                complete_scene_graph, incomplete_scene_graph, query_attribute, given_query, obj_rm, updated = getSceneGraph_data(num_objects, constraint_type_index, env_answers, environment_constraints_dir, args, updated, num_image_per_qa, max_number_of_images_per_qa)
                
                #input(updated)
                """
                      complete_scene_graph = {0: {'color': 'purple', 'material': 'rubber', 'region': '2', 'size': 'medium', 'shape': 'sphere'}, 1: {'color': 'purple', 'material': 'rubber', 'region': '2', 'size': 'large', 'shape': 'sphere'}, 2: {'color': 'purple', 'material': 'rubber', 'region': '0', 'size': 'small', 'shape': 'cylinder'}, 3: {'color': 'yellow', 'material': 'rubber', 'region': '3', 'size': 'small', 'shape': 'cylinder'}, 4: {'color': 'purple', 'material': 'metal', 'region': '0', 'size': 'small', 'shape': 'sphere'}, 5: {'color': 'green', 'material': 'rubber', 'region': '0', 'size': 'small', 'shape': 'sphere'}, 6: {'color': 'green', 'material': 'rubber', 'region': '1', 'size': 'small', 'shape': 'cone'}, 7: {'color': 'gray', 'material': 'rubber', 'region': '1', 'size': 'small', 'shape': 'cone'}, 8: {'color': 'yellow', 'material': 'rubber', 'region': '1', 'size': 'small', 'shape': 'cone'}}
                      """
                print("Scene graph for image ",i, " created!!")



                complete_scene, incomplete_scene = render_scene(args,
                  complete_scene_graph=complete_scene_graph,
                  incomplete_scene_graph=incomplete_scene_graph,
                  image_index=i,
                  complete_scene_image_path=complete_scene_image_path,
                  incomplete_scene_image_path= incomplete_scene_image_path,
                  properties=properties,
                  constraint_type_index=constraint_type_index,
                  phase = args.phase_constraint
                )

                #if complete_scene == None and constraint_type_index == 24:
                #    input(complete_scene_graph)
    
                if complete_scene is not None:
                    #input('scene not none')
                    with open(complete_scene_path, 'w') as f:
                      json.dump(complete_scene, f)
        		
                    with open(incomplete_scene_path, 'w') as f:
                      json.dump(incomplete_scene, f)
                    
                    #GENERATING QUESTION 
                    props = ['color', 'shape', 'size', 'material']
                    query_attribute_index = props.index(query_attribute)		
                    flag_good = False
                    trial = 1
                    tried = []
                    flag_continue = True
                    while(flag_continue): #continue trying
                        if trial != 1:
                            try_next  = False
                            for ind in range(len(num_image_per_qa)):
                                if ind not in tried and num_image_per_qa[ind] < max_number_of_images_per_qa[ind]:
                                    query_attribute_index = ind
                                    try_next = True
                            if not try_next:
                                flag_continue = False
                                break
                            query_attribute = props[query_attribute_index]
                            given_query = chooseGiven(props, query_attribute, 0)
                        
                        for k in range(1, 7):
                        
                            incomplete_scene_path_rel = str(os.path.join(args.incomplete_data_dir, args.scene_dir, args.split))
                            incomplete_scene_main_path = os.path.join(str(path_root), incomplete_scene_path_rel.split("../")[1])
                            environment_constraints_main = os.path.join(str(path_root), 'environment_constraints')
                            question, flag_good = generate_question(args,templates, num_loaded_templates, query_attribute, given_query, obj_rm, complete_scene, complete_scene_path, i, num_questions_per_template_type, max_number_of_questions_per_template, constraint_type_index, incomplete_scene_main_path, environment_constraints_main)
                            if flag_good:
                              break
                              
                            given_query = chooseGiven(props, query_attribute, k)
                            
            			
                        if flag_good:
                            no_question = False
                            flag_continue = False
                            with open(question_path, 'w') as f:
                                json.dump(question, f)
                            #input(question)
                            num_image_per_constraint_type[constraint_type_index]= num_image_per_constraint_type[constraint_type_index] +1
                            num_image_per_qa[query_attribute_index] = num_image_per_qa[query_attribute_index] +1
                            
                        tried.append(query_attribute_index)
                        trial = trial+1
            
                    	
        if not end_of_process:    	
            i = i+1
            print('i updated to:')
            #input(i)
        if args.use_gpu == 1:
          gc.collect()
          
        if (i == args.start_idx + args.render_batch_size)  or end_of_process or i == args.num_images:  #to avoid GPU CUDA overflow!
          if args.phase_constraint != 1:
            #Pickle
		
              num_image_per_constraint_type_file = open(os.path.join(environment_constraints_dir, "num_image_per_constraint_type.pickle"),"wb")
              pickle.dump(num_image_per_constraint_type,num_image_per_constraint_type_file)
              num_image_per_constraint_type_file.close()
              
              num_image_per_qa_file = open(os.path.join(environment_constraints_dir, "num_image_per_qa.pickle"),"wb")
              pickle.dump(num_image_per_qa, num_image_per_qa_file)
              num_image_per_qa_file.close()
              

              possible_num_objects_file = open(os.path.join(environment_constraints_dir, "possible_num_objects.pickle"),"wb")
              pickle.dump(possible_num_objects, possible_num_objects_file)
              possible_num_objects_file.close()
              
              del num_image_per_constraint_type
              del possible_num_objects
              del env_answers
              gc.collect()
              
              updated_file = open(os.path.join(environment_constraints_dir, "updated.obj"),"wb")
              pickle.dump(updated, updated_file)
              updated_file.close()
              
              if end_of_process:
                  print('END OF PROCESS!!!')

          # breaking out of outer while loop - cannot generate more images/env  
          break

  if args.phase_constraint == 1:
      #Pickle env details - give path!!
      env_ans_file = open(os.path.join(environment_constraints_dir,"env_answers.obj"),"wb")
      pickle.dump(env_answers,env_ans_file)
      env_ans_file.close()

      objNum_env_file = open(os.path.join(environment_constraints_dir,"objNum_env.obj"),"wb")
      pickle.dump(objNum_env,objNum_env_file)
      objNum_env_file.close()
      
      print(objNum_env)
      for e in env_answers:
      	print('E:', e, len(env_answers[e]))
       

#------------------------------------------------------------------------------------------------------------------------
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
      
#----------------------------------------------------------------------------------------------------------------------
  

def balance_env_numObj(num_env_per_numObj, max_number_of_env_per_numObj):
  for i in range(len(num_env_per_numObj)):
     if num_env_per_numObj[i] < max_number_of_env_per_numObj:
      return i
  return None  

#-----------------------------------------------------------------------------------------------------------------------
def balance_queryAttribute_numImages(num_images_per_qa, max_number_of_images_per_qa):
  for i in range(len(num_images_per_qa)):
     if num_images_per_qa[i] < max_number_of_images_per_qa[i]:
      return i
  return random.randint(0, 3) 

##---------------------------------------------------------------------------------------------------------------------------
def render_scene(args,
      complete_scene_graph=None,
      incomplete_scene_graph=None,
      image_index=0,
      complete_scene_image_path='render.png',
      incomplete_scene_image_path='render.png',
      properties=None,
      constraint_type_index=None,
      phase=None

  ):

  #input(complete_scene_graph)

  blender_obj = blender.Blender(complete_scene_image_path, 
    args.material_dir, 
    args.base_scene_blendfile, 
    args.width, 
    args.height, 
    args.render_tile_size, 
    args.use_gpu,
    args.render_num_samples,
    args.render_min_bounces, 
    args.render_max_bounces)

   

    

  # This will give ground-truth information about the scene and its objects
  complete_scene_struct = {
       'info': {
          'date': args.date,
          'version': args.version,
          'split': args.split,
          'license': args.license,
       },
      'constraint_type_index': constraint_type_index,
      'image_index': image_index,
      'image_filename': os.path.basename(complete_scene_image_path),
      'objects': [],
      'directions': {}
  }


  plane_behind, plane_left, plane_up = blender_obj.get_plane_direction()
  

  # Save all six axis-aligned directions in the scene struct
  complete_scene_struct['directions']['behind'] = tuple(plane_behind)
  complete_scene_struct['directions']['front'] = tuple(-plane_behind)
  complete_scene_struct['directions']['left'] = tuple(plane_left)
  complete_scene_struct['directions']['right'] = tuple(-plane_left)
  complete_scene_struct['directions']['above'] = tuple(plane_up)
  complete_scene_struct['directions']['below'] = tuple(-plane_up)

  


  loop_counter  = 0
  succeed = False
  # Building a (complete) scene and check the validity and visibility of all the randomly added objects
  while (loop_counter < 100):
    objects, objects_blender_info = add_objects(complete_scene_struct, args, properties, complete_scene_graph)
    objects, blender_objects = get_blender_objects(objects, objects_blender_info, blender_obj)
    all_visible = blender_obj.check_visibility(blender_objects, args.min_pixels_per_object)
 
    if not all_visible:
      # If any of the objects are fully occluded then start over; delete all
      # objects from the scene and place them all again.
      print('Some objects are occluded; replacing objects')
      make_scene_empty(blender_objects)        
      loop_counter = loop_counter + 1
    else:
      succeed = True
      break

  if not succeed:
    return None, None
  else:
    
    complete_scene_struct['objects'] = objects
    complete_scene_struct['relationships'] = scene_info.compute_all_relationships(complete_scene_struct)
    complete_scene_struct['similar'] = scene_info.compute_all_similar(complete_scene_struct)
    #scene_struct['objects_blender_info'] = objects_blender_info    
    
    #if args.phase_constraint == 0:
    #  blender_obj.render()

    
    blender_incomplete_obj = blender.Blender(incomplete_scene_image_path, 
      args.material_dir, 
      args.base_scene_blendfile, 
      args.width, 
      args.height, 
      args.render_tile_size, 
      args.use_gpu,
      args.render_num_samples,
      args.render_min_bounces, 
      args.render_max_bounces) 
    
    
    blender_incomplete_obj.get_plane_direction()

    incomplete_objects, incomplete_blender_info = get_incomplete_scene_info(complete_scene_graph, incomplete_scene_graph, objects, objects_blender_info)
    incomplete_objects, incomplete_blender_objects = get_blender_objects(incomplete_objects, incomplete_blender_info, blender_incomplete_obj)    

    if args.phase_constraint == 0:
      blender_incomplete_obj.render()
    

    incomplete_scene_struct = copy.deepcopy(complete_scene_struct)
    
    
    del incomplete_scene_struct['similar']
    incomplete_scene_struct['image_filename'] = os.path.basename(incomplete_scene_image_path)
    incomplete_scene_struct['objects'] = incomplete_objects
    incomplete_scene_struct['relationships'] = scene_info.compute_all_relationships(incomplete_scene_struct)
    
    del blender_incomplete_obj
    del blender_obj
    gc.collect()
    
    return complete_scene_struct, incomplete_scene_struct


##---------------------------------------------------------------------------------------------------------------------------

def get_incomplete_scene_info(complete_scene_graph, incomplete_scene_graph, objects, blender_objects):
  obj_interest = None
  for obj in complete_scene_graph:
    if obj not in incomplete_scene_graph:
      obj_interest = obj
      break
    else:
      props = incomplete_scene_graph[obj]
      if len(props)<  5:  #len(properties):
        obj_interest = obj
        break

  incomplete_objects = copy.deepcopy(objects)
  incomplete_blender_objects = copy.deepcopy(blender_objects)


  incomplete_objects.pop(obj_interest)
  incomplete_blender_objects.pop(obj_interest)

  return incomplete_objects, incomplete_blender_objects
  

##---------------------------------------------------------------------------------------------------------------------------

def make_scene_empty(blender_objects):
  for obj in blender_objects:
    utils.delete_object(obj)

##---------------------------------------------------------------------------------------------------------------------------
def get_blender_objects(objects, objects_blender_info, blender_obj):
  blender_objects = []
  for index, obj_blender_info in enumerate(objects_blender_info):

    obj, pixel_coords  = blender_obj.add_object(args.shape_dir, 
      obj_blender_info['mat_name'], 
      obj_blender_info['obj_name'], 
      obj_blender_info['r'], 
      obj_blender_info['x'], 
      obj_blender_info['y'], 
      obj_blender_info['theta'],
      obj_blender_info['rgba'])  

    blender_objects.append(obj)
    objects[index]['pixel_coords'] = pixel_coords
    objects[index]['3d_coords'] = tuple(obj.location)
    
    #objects[index]['3d_coords'] = tuple  (obj_blender_info['x'], obj_blender_info['y'], obj.location[2])
    """
    print('---------------------------------------------------')
    print(objects[index]['region'])
    print(obj_blender_info['x'], obj_blender_info['y'])
    print(objects[index]['3d_coords'])
    print('======================================================')    
    """
  return objects, blender_objects
          

##---------------------------------------------------------------------------------------------------------------------------


def get_constraint_types():
  with open(args.constraints_json, 'r') as f_constraints:
    constraint_types = json.load(f_constraints)

  #return constraint_types
  return list(constraint_types.values())


def balance_constraint_type(constraints_types_list, num_image_per_constraint_type, max_number_of_constraints_per_image):
  for i in constraints_types_list:
     if num_image_per_constraint_type[i] < max_number_of_constraints_per_image:
      return i
  return None  

  




##---------------------------------------------------------------------------------------------------------------------------


def get_sorted_list(dict):
    sorted_dictionary = collections.OrderedDict(sorted(dict.items()))
    return list(sorted_dictionary.items())

##---------------------------------------------------------------------------------------------------------------------------


def add_objects(scene_struct, args, properties, complete_scene_graph):
  objects_blender_info = []

  positions = []
  objects = []

  num_objects = len(complete_scene_graph.keys())

  for i in range(num_objects):

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    
    while True:

      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        return add_objects(scene_struct, args, properties, complete_scene_graph)

      
      #x = random.uniform(-3.5, 3.5)
      #y = random.uniform(-3.5, 3.5)

      region_index = complete_scene_graph[i]['region']


      x1 = properties['regions'][region_index]['x'][0]
      x2 = properties['regions'][region_index]['x'][1]
      y1 = properties['regions'][region_index]['y'][0]
      y2 = properties['regions'][region_index]['y'][1]
      x = random.uniform(x1, x2)
      y = random.uniform(y1, y2)
      
      """
      print('region_index: ', region_index)
      print('object: ', i)
      print('ranges: ', x1, x2, y1, y2)
      print('x={} , y={}'.format(x, y))
      print(complete_scene_graph[i]['shape'], complete_scene_graph[i]['color'], complete_scene_graph[i]['size'], complete_scene_graph[i]['material'])
      print('-------------------------------------------')
      """
   

      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break

        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            #margins_good = False
            #break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

      
    shape_name = complete_scene_graph[i]['shape']
    color_name = complete_scene_graph[i]['color']
    size_name = complete_scene_graph[i]['size']
    mat_name = complete_scene_graph[i]['material']

    
    r = properties['size'][size_name]
    rgba = [float(c) / 255.0 for c in properties['color'][color_name]] + [1.0]
    # For cube, adjust the size a bit
    if shape_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    #theta = 360.0

    objects_blender_info.append(
        {'obj_name': properties['shape'][shape_name], 
          'r': r, 'x': x, 'y':y, 'theta': theta, 
          'mat_name': properties['material'][mat_name], 
          'rgba': rgba})
    
    positions.append((x, y, r))
    objects.append({
      'shape': shape_name,
      'size': size_name,
      'material': mat_name,
      'rotation': theta,
      'color': color_name,
      'region': region_index
    })


  return objects, objects_blender_info

##---------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')  

    ## blender --background -noaudio --python render_images.py -- --num_images 1
    ## blender --background -noaudio --python render_images.py -- --num_images 10 --use_gpu 1 --start_idx 0 --num_constraint_types 10
