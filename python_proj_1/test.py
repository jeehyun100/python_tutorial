import os

with open("data.txt",'r+',encoding='UTF-8') as f:
    f.seek(0,1)
    print(f.read())
