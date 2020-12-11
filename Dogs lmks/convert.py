from xml.dom import minidom
import os
import glob
from PIL import Image

def convert_coordinates(size, box):
  dw = 1.0 / size[0]
  dh = 1.0 / size[1]

  # top left width height

  x = box[1] + box[2] / 2.0
  y = box[0] + box[3] / 2.0

  w = box[2]
  h = box[3]

  x = x * dw
  w = w * dw
  y = y * dh
  h = h * dh

  return (x, y, w, h)

def convert_coordinates_lmks(size, coordinates, point_size = 20.0):
  dw = 1.0 / size[0]
  dh = 1.0 / size[1]

  x = coordinates[0]
  y = coordinates[1]
  w = size[0] / point_size
  h = size[0] / point_size

  x = x * dw
  w = w * dw
  y = y * dh
  h = h * dh

  return (x, y, w, h)

labels = {
  "head_top": 4, 
  "lear_base": 2,
  "lear_tip": 3,
  "leye": 0,
  "nose": 1,
  "rear_base": 2,
  "rear_tip": 3, 
  "reye": 0 
}

def main():
  fname = "all_dogs_labeled.xml"
  xmldoc = minidom.parse(fname)
  images = xmldoc.getElementsByTagName('image')
  for image in images:
    fname = image.attributes["file"].value

    img = Image.open(fname)    
    size = img.size

    parts = image.getElementsByTagName('part')

    f = open(fname[:-3] + "txt", "w+")

    for part in parts:
      x = int(part.attributes["x"].value)
      y = int(part.attributes["y"].value)
      coordinates = (x, y)

      name = part.attributes["name"].value

      num = labels[name]

      if(num > 1):
        coordinates = convert_coordinates_lmks(size, coordinates, point_size=20)
      else:      
        coordinates = convert_coordinates_lmks(size, coordinates, point_size=20)
      
      f.write(str(labels[name]) + " " + " ".join([("%.6f" % a) for a in coordinates]) + '\n')

main()