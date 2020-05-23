from os import listdir
from os.path import isfile, join
import shutil
import random

numfiles = len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))
print("numfiles: "+str(numfiles))
print("moving 60% to train")
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))*0.60)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\train\pis\\"+choice)
print("--------moved to train, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))))
for i in range(int(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))*0.50)):
    choice = random.choice(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))
    print(choice)
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis\\"+choice, r"C:\Users\tony\Desktop\gun_dataset\dataset\test\pis\\"+choice)
#print((listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis")))
print("--------moved to test, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))))
for i in listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"):
    shutil.move(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis\\"+i, r"C:\Users\tony\Desktop\gun_dataset\dataset\val\pis\\"+i)
print("--------moved to val, {} files remaining".format(len(listdir(r"C:\Users\tony\Desktop\gun_dataset\img_adj_pis"))))
