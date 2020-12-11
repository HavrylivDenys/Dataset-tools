from xml.dom import minidom
import os
import glob

label_list = {}
label_list["cat"] = 0
label_list["dog"] = 1

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)

for fname in glob.glob("./xmls/*.xml"):

  xmldoc = minidom.parse(fname)

  fname_out = (fname[7:-4]+'.txt')

  path = "new_text"

  fname_out = os.path.join(path, fname_out)

  with open(fname_out, "w") as f:
    itemlist = xmldoc.getElementsByTagName('object')
    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    for item in itemlist:
      classid =  (item.getElementsByTagName('name')[0]).firstChild.data
      if classid in label_list:
        label = str(label_list[classid])
      else:
        label = "-1"
        print ("warning: label '%s' not in look-up table" % classid)
      xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
      ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
      xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
      ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
      b = (float(xmin), float(xmax), float(ymin), float(ymax))
      bb = convert_coordinates((width,height), b)

      f.write(label + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

    print ("wrote %s" % fname_out)

# mv -Path ".\xmls\*.txt" -Destination ".\text"