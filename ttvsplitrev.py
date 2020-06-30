from os import listdir
from os.path import isfile, join
import shutil
import random

numfiles = len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg\\"))
print(numfiles)
print("numfiles: "+str(numfiles))
print("moving 60% to train")
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))*0.60)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\train\smg\\"+choice)
print("--------moved to train, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))))
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))*0.50)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\test\smg\\"+choice)
#print((listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg")))
print("--------moved to test, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))))
for i in listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"):
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg\\"+i, r"C:\Users\tony\Desktop\gun_dataset\dataset\val\smg\\"+i)
print("--------moved to val, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_smg"))))
