import requests
from bs4 import BeautifulSoup
import pandas as pd

linkpage_pistol=requests.get('http://www.imfdb.org/wiki/Category:Pistol')
linkpage_revolver=requests.get('http://www.imfdb.org/wiki/Category:Revolver')
page_prefix = "http://www.imfdb.org"
linksoup = BeautifulSoup(linkpage_pistol.content)

content = linksoup.find("div",{"id":"mw-content-text"})
tables = content.find_all("table")
num = 0
for table in tables:
    print("---------------")
    #print(table)
    galleryboxes = table.find_all("li", class_="gallerybox")
    if galleryboxes:
        #print(" i have gallery")
        for entry in galleryboxes:
            image = entry.find("img").get('src')
            #print(image)
            print(page_prefix+image)
            pic = requests.get(page_prefix+image)
            with open(r"C:\Users\tony\Desktop\gun_dataset\img_pis\{}.jpg".format(num), 'wb') as out_file:
                out_file.write(pic.content)

            #print("pic wrote")
            #pic.save("\img\ "+num+".jpg")
            #print("--")
            image_string = entry.find("div", class_="gallerytext").p.get_text()
            with open(r"C:\Users\tony\Desktop\gun_dataset\text_pis\{}.txt".format(num), 'w') as out_file:
                out_file.write("Pistol"+'\n'+image_string)
            print(image_string)

            print()
            num=num+1
    else:
        print("i dont have gallery")
    #num=num+1
#galleryboxes = tables.find(_class="gallerybox")

linksoup = BeautifulSoup(linkpage_revolver.content)

content = linksoup.find("div",{"id":"mw-content-text"})
tables = content.find_all("table")
num = 0
for table in tables:
    print("---------------")
    #print(table)
    galleryboxes = table.find_all("li", class_="gallerybox")
    if galleryboxes:
        #print(" i have gallery")
        for entry in galleryboxes:
            image = entry.find("img").get('src')
            #print(image)
            print(page_prefix+image)
            pic = requests.get(page_prefix+image)
            with open(r"C:\Users\tony\Desktop\gun_dataset\img_rev\{}.jpg".format(num), 'wb') as out_file:
                out_file.write(pic.content)

            #print("pic wrote")
            #pic.save("\img\ "+num+".jpg")
            #print("--")
            image_string = entry.find("div", class_="gallerytext").p.get_text()
            with open(r"C:\Users\tony\Desktop\gun_dataset\text_rev\{}.txt".format(num), 'w') as out_file:
                out_file.write("Revolver"+'\n'+image_string)
            print(image_string)

            print()
            num=num+1
    else:
        print("i dont have gallery")
