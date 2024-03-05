import sys, random, tempfile, os
from collections import Counter



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


class Blender():
  def __init__(self, image_path, material_dir, base_scene_blendfile, width, height, render_tile_size, use_gpu, render_num_samples, render_min_bounces, render_max_bounces):
    self.initialize(image_path, material_dir, base_scene_blendfile, width, height, render_tile_size, use_gpu, render_num_samples, render_min_bounces, render_max_bounces)


  def initialize(self, image_path, material_dir, base_scene_blendfile, width, height, render_tile_size, use_gpu, render_num_samples, render_min_bounces, render_max_bounces):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=base_scene_blendfile)

    # Load materials
    utils.load_materials(material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = image_path
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
      # Blender changed the API for enabling CUDA at some point
      if bpy.app.version < (2, 78, 0):
        bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        bpy.context.user_preferences.system.compute_device = 'CUDA_0'
      else:
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
      bpy.context.scene.cycles.device = 'GPU'


    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    self.plane = bpy.context.object
    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    self.camera = bpy.data.objects['Camera']

    #self.set_camera_location(location=location, camera_jitter=camera_jitter)

  """
  def get_camera_location(self):
      return bpy.data.objects['Camera'].location
  

  def set_camera_location(self, camera_jitter, location):

    if location is not None:
      bpy.data.objects['Camera'].location = location
    else:
      # Add random jitter to camera position
      if camera_jitter > 0:
        for i in range(3):
          bpy.data.objects['Camera'].location[i] += Blender.rand(camera_jitter)     
  """
  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  def get_plane_direction(self):
    plane_normal = self.plane.data.vertices[0].normal
    cam_behind = self.camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = self.camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = self.camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()
    

    self.delete_object(self.plane)       
    #print(plane_behind, plane_left, plane_up)
    #input('here')
    return plane_behind, plane_left, plane_up

  def render(self):
    while True:
      try:
        bpy.ops.render.render(write_still=True)
        #bpy.ops.wm.save_as_mainfile(filepath='/home/test.blend')
        
        break
      except Exception as e:
        print(e)

  
  def delete_object(self, obj):
    utils.delete_object(obj) 
  
  def delete_plane(self):
    self.delete_object(self.plane)


  def add_object(self, shape_dir, mat_name, obj_name, r, x, y, theta, rgba):
    # Actually add the object to the scene
    utils.add_object(shape_dir, obj_name, r, (x, y), theta)
    obj = bpy.context.object
    utils.add_material(mat_name, Color=rgba)
    
    location = Vector((x, y, obj.location[2]))
    obj.location = location
    pixel_coords = utils.get_camera_coords(self.camera, obj.location)
    return obj, pixel_coords



  def check_visibility(self, blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = self.render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                          for i in range(0, len(p), 4))
    
    os.close(f)
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
      return False
    for _, count in color_count.most_common():
      if count < min_pixels_per_object:
        return False
    return True

  def render_shadeless(self, blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render

    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)
    
    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
      old_materials.append(obj.data.materials[0])
      bpy.ops.material.new()
      mat = bpy.data.materials['Material']
      mat.name = 'Material_%d' % i
      while True:
        r, g, b = [random.random() for _ in range(3)]
        if (r, g, b) not in object_colors: break
      object_colors.add((r, g, b))
      mat.diffuse_color = [r, g, b]
      mat.use_shadeless = True
      obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)
    

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
      obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    

    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)
 
  
    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors
