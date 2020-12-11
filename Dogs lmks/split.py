import os
from random import seed
from random import random
from os import makedirs
from os import listdir
from os import path
from shutil import copy
import glob
from PIL import Image
import numpy as np


subdirs = ['dogImages/test/', 'dogImages/obj/']

for subdir in subdirs:
  makedirs(subdir, exist_ok=True)

# seed random number generator
seed(2)
# define ratio of pictures to use for validation
val_ratio = 0.1
counter = 0
file_list = []

for file in glob.glob(".\dogImages/**/*.jpg"):
  try:
    img=Image.open(file)
    if(img.format == "JPEG"):     
      if(path.exists(file[:-4] + ".txt")):
        file_list.append(file[12:])
        print(os.path.basename(file[12:]))
  except(IOError,SyntaxError) as e:
      print('Bad file  :  '+file)

np.random.shuffle(file_list)

for file in file_list:

  dir_base = "dogImages"
  test_dir = os.path.join(dir_base, 'test')
  obj_dir = os.path.join(dir_base, 'obj')

  counter = counter + 1  

  if random() < val_ratio:
    dst_dir = test_dir
  else:
    dst_dir = obj_dir

  file_name = os.path.basename(file)

  path = os.path.join(dir_base, file)
  dst_path = os.path.join(dst_dir, file_name)
  copy(path, dst_path)  
  txt_file = file[:-4] + ".txt"
  txt_file_name = file_name[:-4] + ".txt"
  path = os.path.join(dir_base, txt_file)
  dst_path = os.path.join(dst_dir, txt_file_name)
  copy(path, dst_path)  
  print(counter, file, txt_file)
