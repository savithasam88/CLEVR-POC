import random

from  constraint import *

class Region:
    def __init__(self, x_range=None, y_range=None, constraints=None, properties=None):

        self.x_range = x_range
        self.y_range = y_range

        #set CSP settings related to the region 
        self.set_constraints(constraints, properties)
    

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
        #print(self.solutions)
       
    def get_all_solutions(self):
        return self.solutions
        
    def contains(self, x, y):
        return self.x_range[0] <= x and x <= self.x_range[1] and self.y_range[0] <= y and y <= self.y_range[1]

    def get_region_features(self):
        features = random.choice(self.solutions)
        shape = features['shape']
        color = features['color']
        size = features['size']
        material = features['material']
        return shape, color, size, material

         
    
