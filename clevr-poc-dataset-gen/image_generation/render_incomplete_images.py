# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from __future__ import print_function
from ast import Not
import imp
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import collections 

from pathlib import Path
import sys
from xml.etree.ElementTree import register_namespace
import shutil

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
#print(sys.path)
from image_generation import scene_info, blender
from generate_dataset import parser


"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape, compatible with given constraints. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

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

"""

parser = argparse.ArgumentParser()

parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")

parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=320, type=int,
    help="The height (in pixels) for the rendered images")
  
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")    
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")           

parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")            

parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")

parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")

#Input
parser.add_argument('--input_scene_file', default='../output/complete/scenes/all_scenes.json',
    help="Path to write a single JSON file containing all complete scene information")
    
parser.add_argument('--input_questions_file', default='../output/CLEVR_questions.json',
    help="The output file to write containing generated questions")

#Output
parser.add_argument('--output_incomplete_image_dir', default='../output/incomplete/images/',
    help="The directory where output incomplete images will be stored. It will be " +
         "created if it does not exist.")

parser.add_argument('--output_incomplete_scene_dir', default='../output/incomplete/scenes/',
    help="The directory where output JSON incomplete scene structures will be stored. " +
         "It will be created if it does not exist.")

parser.add_argument('--output_incomplete_question_dir', default='../output/incomplete/questions/',
    help="The directory where output JSON incomplete questions will be stored. " +
         "It will be created if it does not exist.")

parser.add_argument('--output_training_file', default='training.json',
    help="File name related to training data")
parser.add_argument('--output_validation_file', default='validation.json',
    help="File name related to validation data")
parser.add_argument('--output_testing_file', default='testing.json',
    help="File name related to testing data")        

parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")

parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")

"""
#----------------------------------------------------------------------------------------
def render_incomplete_scene(args,
            objects,
            objects_blender_info,
            image_path,
            image_index,
            object_of_interest_index,
            directions):

    blender_obj = blender.Blender(image_path, 
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
    incomplete_scene_struct = {
        'image_filename': os.path.basename(image_path),
        'image_index': image_index,
        'object_of_interest_index': object_of_interest_index,
        'objects': objects,
        'directions': directions,
        'relationships': {},
    }

    incomplete_scene_struct['relationships'] = scene_info.compute_all_relationships(incomplete_scene_struct)
    blender_obj.delete_plane()

    objects, blender_objects = get_blender_objects(objects, objects_blender_info, blender_obj)
    
    
    blender_obj.render()
    return incomplete_scene_struct

    
#----------------------------------------------------------------------------------------         

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

  return objects, blender_objects


#----------------------------------------------------------------------------------------------------------------------------

def main(args):
    
    image_dir = os.path.join(args.incomplete_data_dir, args.image_dir, args.split)
    scene_dir = os.path.join(args.incomplete_data_dir, args.scene_dir)
    question_dir = os.path.join(args.incomplete_data_dir, args.question_dir)

    if args.start_question_idx == 0:
      if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
    temp_dir = os.path.join(args.temp_dir)


    if not os.path.isdir(image_dir):
      os.makedirs(image_dir) 
    if not os.path.isdir(scene_dir):
      os.makedirs(scene_dir) 
    if not os.path.isdir(question_dir):
      os.makedirs(question_dir)             
    
    if not os.path.isdir(temp_dir):
      os.makedirs(temp_dir) 


    with open(os.path.join(args.incomplete_data_dir, args.question_dir, args.split + '.json'), 'r') as f:
        questions_info = json.load(f)
    
  

    with open(os.path.join(args.complete_data_dir, args.scene_dir, args.split + '.json'), 'r') as f:
        complete_scenes_info = json.load(f)
        complete_scenes_by_name = {}
        for scene in complete_scenes_info['scenes']:
            complete_scenes_by_name[scene['image_filename']] = scene



    all_incomplete_scenes = get_already_rendered_scenes(split=args.split, scene_dir=scene_dir)
    #complete_file_names = {}
    complete_file_names =read_temp_files(temp_dir=temp_dir)

    questions = questions_info['questions']
    
    i = args.start_question_idx
    #for i, question in enumerate(questions):  # we need as number of questions as incomplete images to generate
    while i < len(questions):
      question = questions[i]
   
      # Extracting the index of the object going to be removed from the scene
      object_of_interest_index = question['program'][-2]['_output']        
      
      objects = complete_scenes_by_name[question['image_filename']]['objects']
      objects_blender_info = complete_scenes_by_name[question['image_filename']]['objects_blender_info']
      directions = complete_scenes_by_name[question['image_filename']]['directions']
      
  
      objects = [o for j, o in enumerate(objects) if j != object_of_interest_index]
      objects_blender_info = [o for j, o in enumerate(objects_blender_info) if j != object_of_interest_index]


      print('------------------------------------')
      # Extracting the complete image and scene file name and the current incomplete image file name
      complete_image_file_name = question['image_filename']
      original_file_name = complete_image_file_name.split('.')[0]
      
      index =  complete_file_names.get(original_file_name)
      index = 0 if index is None else (index+1)
      complete_file_names[original_file_name] = index
      image_file_name =  original_file_name + '_' + str(index) + '.png'
      image_path = os.path.join(image_dir, image_file_name)
      
      
      incomplete_scene = render_incomplete_scene(args=args,
          objects=objects,
          objects_blender_info=objects_blender_info,
          image_path=image_path,
          image_index=i,
          object_of_interest_index=object_of_interest_index,
          directions=directions)      

      incomplete_scene['complete_image_file_name'] = complete_image_file_name
      incomplete_scene['complete_image_index'] = question['image_index']
      incomplete_scene['constraint_type'] = complete_scenes_by_name[question['image_filename']]['constraint_type']

      all_incomplete_scenes.append(incomplete_scene)  

      question['image_filename'] = image_file_name
      question['complete_image_index'] = question['image_index']
      question['image_index'] = i
      question['constraint_type'] = incomplete_scene['constraint_type']
    

      i += 1
      if i == args.start_question_idx + args.render_batch_size:  #to avoid GPU CUDA overflow!
        break

      

    questions_info['questions'] = questions
    with open(os.path.join(question_dir, args.split + '.json'), 'w') as f:
      json.dump(questions_info, f)  
    


    output = {
      'info': {
        'date': args.date,
        'version': args.version,
        'split': args.split,
        'license': args.license,
      },
      'scenes': all_incomplete_scenes
    }
    with open(os.path.join(scene_dir, args.split + '.json'), 'w') as f:
      json.dump(output, f)  

    #temporarily saving files to control indexing of incomplete scenes
    save_temp_files(temp_dir, complete_file_names)





def get_already_rendered_scenes(split, scene_dir):
  if os.path.exists(os.path.join(scene_dir, split + '.json')):
    with open(os.path.join(scene_dir, args.split + '.json'), 'r') as f:
      data = json.load(f)
    scenes = data['scenes']
  else:
    scenes = []
  
  return scenes

def read_temp_files(temp_dir):
  file_name = os.path.join(temp_dir, 'complete_file_names' + '.json')
  if os.path.exists(file_name):
    with open(file_name, 'r') as f:
      complete_file_names = json.load(f)
  else:
    complete_file_names = {}
  
  return complete_file_names



def save_temp_files(temp_dir, complete_file_names):
  with open(os.path.join(temp_dir, 'complete_file_names' + '.json'), 'w') as f:
    json.dump(complete_file_names, f)

#----------------------------------------------------------------------------------------


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
    print('blender --background --python generate_incomplete_scenes.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

    ## blender --background -noaudio --python render_incomplete_images.py
    ## blender --background -noaudio --python render_incomplete_images.py --  --render_batch_size 2 --start_question_idx 32 --use_gpu 0

  

  