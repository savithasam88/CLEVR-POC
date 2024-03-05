import os, argparse, sys
from datetime import datetime as dt
from pathlib import Path


parser = argparse.ArgumentParser()


# ---------------------------------------------------------------------------------------------------------------
## PATHS

parser.add_argument('--base_scene_blendfile', default='../data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
      
parser.add_argument('--shape_dir', default='../data/shapes',
    help="Directory where .blend files for object models are stored")

parser.add_argument('--material_dir', default='../data/materials',
    help="Directory where .blend files for materials are stored")

parser.add_argument('--properties_json', default='../data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")

parser.add_argument('--constraints_json', default='../data/constraints.json',
    help="Optional path to a JSON file containing constrainst in a scene.")



parser.add_argument('--complete_data_dir', default='../output/complete/',
    help="The directory where output data about complete images/scenes will be stored. It will be " +
         "created if it does not exist.")

parser.add_argument('--image_dir', default='images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")

parser.add_argument('--scene_dir', default='scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")

parser.add_argument('--scenes_file', default='scenes.json',
    help="Path to write a single JSON file containing all scenes information")

parser.add_argument('--incomplete_data_dir', default='../output/incomplete/',
    help="The directory where output data about incomplete images/scenes/questions will be stored. It will be " +
         "created if it does not exist.")

parser.add_argument('--question_dir', default='questions/',
    help="The directory where output JSON incomplete questions will be stored. " +
         "It will be created if it does not exist.")


parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")

parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")

parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")


parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='training',
    help="Name of the split for which we are rendering (training, validation, testing)")


# Settings for objects
parser.add_argument('--min_objects', default=5, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=9, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.3, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.42, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=80, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

parser.add_argument('--template_dir', default='CLEVR_POC_templates',
    help="Directory containing JSON templates for questions")

parser.add_argument('--render_batch_size', default=500, type=int,
    help="the batch size indicates how often to restrat rendering")

parser.add_argument('--num_constraint_types', default=200, type=int,
    help="The number of environments (constraint types) in each blender batch size of instances")

parser.add_argument('--constraint_template_path', default='../image_generation/ConstraintTemplates/constraint_templates.txt',
    help="File containing templates for constraints")

parser.add_argument('--general_constraints_path', default='../data/general_constraints.txt',
    help="File containing general constraints")

parser.add_argument('--environment_constraints_dir', default='../environment_constraints/',
    help="The directory where constraints about environments will be placed.")    

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=320, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")




# ------------------------------------------------------------
#Question generation
# Inputs


parser.add_argument('--start_question_idx', default=0, type=int,
    help="The question index at which to start for rendering incomplete images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")


parser.add_argument('--metadata_file', default='metadata.json',
    help="JSON file containing metadata about functions")

parser.add_argument('--synonyms_json', default='synonyms.json',
    help="JSON file defining synonyms for parameter values")


parser.add_argument('--temp_dir', default='render_temp',
    help="Directory containing temp JSON files required to control indexing of incomplete scenes")

# Control which and how many images to process
parser.add_argument('--num_templates', default=28, type=int,
    help="The number of question templates")

parser.add_argument('--scene_start_idx', default=0, type=int,
    help="The image at which to start generating questions; this allows " +
         "question generation to be split across many workers")
parser.add_argument('--num_scenes', default=0, type=int,
    help="The number of images for which to generate questions. Setting to 0 " +
         "generates questions for all scenes in the input file starting from " +
         "--scene_start_idx")

# Control the number of questions per image; we will attempt to generate
# templates_per_image * instances_per_template questions per image.


parser.add_argument('--templates_per_image', default=1, type=int,
    help="The number of different templates that should be instantiated " +
         "on each image")
parser.add_argument('--instances_per_template', default=1, type=int,
    help="The number of times each template should be instantiated on an image")

# Misc
parser.add_argument('--reset_counts_every', default=250, type=int,
    help="How often to reset template and answer counts. Higher values will " +
         "result in flatter distributions over templates and answers, but " +
         "will result in longer runtimes.")
parser.add_argument('--verbose', action='store_true',
    help="Print more verbose output")
parser.add_argument('--time_dfs', action='store_true',
    help="Time each depth-first search; must be given with --verbose")
parser.add_argument('--profile', action='store_true',
    help="If given then run inside cProfile")
# args = parser.parse_args()


parser.add_argument('--phase_constraint', default=1, type=int,
    help="Indicating whether we are in constraint generation phase or data generation phase")
# args = parser.parse_args()


if __name__ == '__main__':
    os.system('cd image_generation')
    os.system('blender --background -noaudio --python image_generation/render_image.py -- --num_images 1')
    #print(Path(__file__).parents[0])
    #path_root = Path(__file__).parents[0]
    #sys.path.append(str(path_root))

    #argv = utils.extract_args()
    #args = parser.parse_args(argv)
    #test.start(args)
