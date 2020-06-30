import imagesize
import sys
import glob
from os import listdir
from os.path import isfile, join
from pathlib import Path

maxheight = 0
maxwidth = 0

if len(sys.argv)!=2:
    print("Usage: python checkmaxsize.py path_extension")
    exit()

pictures = sorted(Path(Path.cwd()/sys.argv[1]).glob('*'))


for f in pictures:
    width, height = imagesize.get(f)
    print("w: "+str(width)+" h: "+str(height))
    if width>maxwidth:
        maxwidth=width
    if height>maxheight:
        maxheight=width
print("maxw: "+str(maxwidth)+" maxh: "+str(maxheight))
