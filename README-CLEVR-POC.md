1. Change the directory to the following:
CLEVR-POC/clevr-poc-dataset-gen/image_generation/render_images.py

2. Environment generation - In order to generate 30 environments that allow scenes with number of objects ranging from 5 to 9, run the following command: 
		blender --background -noaudio --python render_images.py -- --use_gpu 1 --phase_constraint 1  --num_constraint_types 30  --min_objects 5 --max_objects 9

3. The 30 environments generated can be found at: CLEVR-POC/clevr-poc-dataset-gen/environment_constraints

4. Data Generation
	4.1   Generating training data (that is images+questions) with number of instances =  500, where each image is belonging to one of the 30 environments generated in the previous step, run the following command: 
		blender --background -noaudio --python render_images.py -- --use_gpu 1 --phase_constraint 0 --num_constraint_types 30 --split 'training' --num_images 500

	4.2 Generating validation data (that is images+questions) with number of instances =  50, where each image is belonging to one of the 30 environments generated in the previous step, run the following command:
		blender --background -noaudio --python render_images.py -- --use_gpu 1 --phase_constraint 0 --num_constraint_types 30 --split 'validation' --num_images 50

	4.3 Generating testing data (that is images+questions) with number of instances =  50, where each image is belonging to one of the 30 environments generated in the previous step, run the following command:
		blender --background -noaudio --python render_images.py -- --use_gpu 1 --phase_constraint 0 --num_constraint_types 30 --split 'testing' --num_images 50

5. The data generated can be found at: CLEVR-POC/clevr-poc-dataset-gen/output-500
 

