import json
import os
import shutil
# import random


path_to_json = './val/'

json_files = [file for file in os.listdir(path_to_json) if file.endswith('.json')]

labels = {}
for json_file in json_files:
    print(json_file)
    print(json_files)
    with open(path_to_json+json_file) as fi:
        person = json_file.replace(".json","")
        labels[person] = {k:v for k,v in json.load(fi).items() if len(v['regions']) > 0}

#os.makedirs('dataset')
#os.makedirs('dataset/mergedtrain')
os.makedirs('dataset/mergedval')


train_labels = {}
#val_labels = {}

for p in labels.keys():
    for k,v in labels[p].items():
        # print(p+'_'+v['filename'])
        name = v['filename'] 
        v['filename'] = p+'_'+name
        # if random.random() < 0.8:
        train_labels[p+'_'+k] = v
        shutil.copy("./val/"+name,"dataset/mergedval/"+p+"_"+name)
        # else:
           # val_labels[p+'_'+k] = v
           # shutil.copy("C:/Users/Caio Albuquerque/Desktop/samples/"+name,"dataset/val/"+p+"_"+name)
            
with open("./dataset/mergedval/via_region_water_data.json","w") as fo:
    json.dump(train_labels,fo)

#with open("dataset/mergedval/via_region_water_data.json","w") as fo:
    #json.dump(val_labels,fo)

print('rodou')