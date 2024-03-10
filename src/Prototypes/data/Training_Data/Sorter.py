import os
import shutil


#print('Welcome to the file sorter program')
#numOfCats= int(input('Please how many categories would you like to sort the files into: '))
#int(numOfCats)
#count = 0
#while count < numOfCats:
#    input('Please enter the name of category ' + str(count+1) + ' : ')
    

DictOfCats = {'report':False,'thunder':False,'rain':False,'wind':False,'other':False}

folders=[]

directory = "C:/Users/User/Downloads/"
        
destination ='C:\Users\User\Documents\GitHub\Project-Echo\Training_Data\Data'
for filename in os.listdir(destination):
    filepath = destination+filename
    if os.path.isfile(filepath):
        print(filename + ' is a file')
    elif os.path.isdir(destination+filename):
        print(filename + ' is a folder')
        filename=filename.lower()
        folders.append(filename)
    else:
        print('ERROR: something went wrong')
    
for folder in folders:
    for item in DictOfCats:
        if folder == item:
            DictOfCats[item] = True

count=0
for item in DictOfCats:
    if DictOfCats[item]:
        pass
        count=count+1
    else:
        path = os.path.join(destination, item)
        os.mkdir(path)
        count=count+1

for item in DictOfCats:
    for filename in os.listdir(directory):
        if item in filename:
            filepath=directory+filename
            desFilePath=destination+item+'/'+filename
            shutil.move(filepath, desFilePath)