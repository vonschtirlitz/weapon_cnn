from os import listdir
from os.path import isfile, join
import shutil
import random

numfiles = len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))
print("numfiles: "+str(numfiles))
print("moving 60% to train")
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))*0.60)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\train\rev\\"+choice)
print("--------moved to train, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))))
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))*0.50)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\test\rev\\"+choice)
#print((listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev")))
print("--------moved to test, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))))
for i in listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"):
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev\\"+i, r"C:\Users\tony\Desktop\gun_dataset\dataset\val\rev\\"+i)
print("--------moved to val, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_rev"))))
