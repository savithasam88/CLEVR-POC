import os, math, gc
## sizes are w.r.t incomplete dataset i.e., number of incomplete scenes

def remove_tem_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print("The file has been deleted successfully")
    else:
        print("The file does not exist!")

num_constraint_types = 30
training_size = 500
testing_size = int(math.ceil(training_size/10))
validation_size = int(math.ceil(training_size/10))

use_gpu=1
render_batch_size=40
start_idx=0

dataset_names=['training', 'testing', 'validation']
dataset_sizes=[training_size, testing_size, validation_size]




os.chdir('image_generation')
os.system('echo $PWD')
for i, dataset in enumerate(dataset_names):
    
    num_images = int(math.ceil(dataset_sizes[i]))
    print(num_images)
    start_idx=0
    while(True):
        gc.collect()
        os.system('blender --background -noaudio --python render_images.py -- --num_images ' + str(num_images) + ' --split ' + dataset_names[i] + ' --use_gpu ' + str(use_gpu) + ' --render_batch_size ' + str(render_batch_size) + ' --start_idx ' + str(start_idx) + ' --num_constraint_types ' + str(num_constraint_types) + ' --phase_constraint 0')
        
        start_idx += render_batch_size
        if start_idx >= num_images:
            break
        print('complete: start_index_', start_idx)
    	

	
path = '../environment_constraints'
#remove_tem_file(os.path.join(path, 'env_answers_updated.obj'))
remove_tem_file(os.path.join(path, 'updated.obj'))
remove_tem_file(os.path.join(path, 'num_image_per_constraint_type.pickle'))
remove_tem_file(os.path.join(path, 'possible_num_objects.pickle'))


    
    

