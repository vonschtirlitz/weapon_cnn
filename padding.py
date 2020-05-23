import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(r"C:\Users\tony\Desktop\gun_dataset\img_rev") if isfile(join(r"C:\Users\tony\Desktop\gun_dataset\img_rev", f))]
#print(onlyfiles)
for f in onlyfiles:
    img = cv2.imread(r"C:\Users\tony\Desktop\gun_dataset\img_rev\\"+f)
    ht, wd, cc= img.shape
    ww = 250
    hh = 250
    color = (255,255,255)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    result[yy:yy+ht, xx:xx+wd] = img
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev\\"+f, result)
