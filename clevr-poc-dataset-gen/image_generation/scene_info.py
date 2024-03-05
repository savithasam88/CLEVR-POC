import random, sys
#sys.path.append("/home/code/clevr-poc/clevr-dataset-gen-main/image_generation")

from  constraint import *

class Region:
    def __init__(self, x_range=None, y_range=None, index=None, constraints=None, properties=None):

        self.x_range = x_range
        self.y_range = y_range
        self.index = index

        self.solutions = constraints

        ##set CSP settings related to the region 
        #self.set_constraints(constraints, properties)
    
    def get_all_solutions(self):
        return self.solutions


    def get_index(self):
        return self.index


    """
    def set_constraints(self, constraints, properties):
               
        #CSP problem
        problem = Problem()

        #CSP variables
        for key,value in properties.items():
            problem.addVariable(key, [*value.keys()])
        
        def generate_constrained_func(c):
            def constraint_func(a, b):
                return ([a, b] in c)
            return constraint_func

        for key in constraints:
            variable_names = key.split("_")
            if len(variable_names) == 1:
                problem.addConstraint(InSetConstraint(constraints[key]), [variable_names[0]])
            elif len(variable_names) == 2:
                problem.addConstraint(generate_constrained_func(constraints[key]), (variable_names[0], variable_names[1]))


        self.solutions = problem.getSolutions()
    
       
    def get_all_solutions(self):
        return self.solutions
    """
    def contains(self, x, y):
        return self.x_range[0] <= x and x <= self.x_range[1] and self.y_range[0] <= y and y <= self.y_range[1]

    def get_region_features(self):
        features = random.choice(self.solutions)
        shape = features['shape']
        color = features['color']
        size = features['size']
        material = features['material']
        return shape, color, size, material

         
    

           
def find_region(regions, x, y):

    for region in regions:
        if region.contains(x, y):
            return region
    return None


##---------------------------------------------------------------------------------------------------------------------------

def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships
"""


  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
        all_relationships[name].append(set())

  for i, obj1 in enumerate(scene_struct['objects']):
    coords1 = obj1['3d_coords']
    related = set()
    for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        
        if coords1[0] > coords2[0]:
            all_relationships['left'][i].add(j)
        if coords1[0] < coords2[0]:
            all_relationships['right'][i].add(j)


        if coords1[1] > coords2[1]:
            all_relationships['front'][i].add(j)
        if coords1[1] < coords2[1]:
            all_relationships['behind'][i].add(j)            

  for name, relations in all_relationships.items():
    for i, obj1 in enumerate(scene_struct['objects']):
        all_relationships[name][i] = sorted(list(all_relationships[name][i]))

  return all_relationships
  

def compute_all_similar(scene_struct):
    all_similar = {}
    for p in ['color', 'size', 'material', 'shape']:
        sim_p = []
        objects = scene_struct['objects']
        for i, obj in enumerate(objects):
            obj_sim_p = []
            for j, obj_other in enumerate(objects):
                if i == j:
                    continue
                else:
                    if obj[p]==obj_other[p]:
                        obj_sim_p.append(j)
            sim_p.append(obj_sim_p)
        all_similar[p] = sim_p
    return all_similar
    
