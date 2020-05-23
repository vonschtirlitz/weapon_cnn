import imagesize
from os import listdir
from os.path import isfile, join

maxheight = 0
maxwidth = 0

onlyfiles = [f for f in listdir(r"C:\Users\tony\Desktop\gun_dataset\img_rev") if isfile(join(r"C:\Users\tony\Desktop\gun_dataset\img_rev", f))]
#print(onlyfiles)
for f in onlyfiles:
    width, height = imagesize.get(r"C:\Users\tony\Desktop\gun_dataset\img_rev\\"+f)
    print("w: "+str(width)+" h: "+str(height))
    if width>maxwidth:
        maxwidth=width
    if height>maxheight:
        maxheight=width
print("maxw: "+str(maxwidth)+" maxh: "+str(maxheight))
