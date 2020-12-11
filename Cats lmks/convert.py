import os
import glob
from os import makedirs
from PIL import Image
from shutil import copy
import numpy as np

def checkPoint(point, size):
  x, y = p[0], p[1]
  if(x < 0):
    x = 0
  elif(x > size[0]):
    x = size[0]
  if(y < 0):
    y = 0
  elif(y > size[1]):
    y = size[1]
  return (x, y)

def convert_coordinates(size, point, point_size = 20.0):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = point[0]
    y = point[1]

    w = size[0] / point_size
    h = size[1] / point_size

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)


for file in glob.glob("*/*.jpg"):

  img = Image.open(file)
  w, h = img.size

  annotation_file = file + '.cat'

  f = open(annotation_file, "r+")
  data = f.read().split(" ")[1:-1]
  points = np.array([data[i: i+2] for i in range(0, len(data), 2)], dtype=np.float64)
  
  new_file = annotation_file[7:-8] + ".txt"
  dir_path = 'text'
  makedirs(dir_path, exist_ok=True)
  path = os.path.join(dir_path, new_file)

  f2 = open(path, "w+")

  labels = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 3,
    8: 2
  }

  for i, p in enumerate(points):
    p = checkPoint(p, (w, h))
    if(i > 2):
      bb = convert_coordinates((w, h), p, point_size = 20)
    else:
      bb = convert_coordinates((w, h), p, point_size = 20)
    f2.write(str(labels[i]) + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
  
  f2.close()

  dst_fname = os.path.join(dir_path, file[7:])
  copy(file, dst_fname)