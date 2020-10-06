#!/usr/bin/env python3

import json, codecs

import numpy as np

# instantiate an empty dict
team = {}

# add a team member
team['tux'] = {'health': 23, 'level': 4}
team['beastie'] = {'health': 13, 'level': 6}
team['konqi'] = {'health': 18, 'level': 7}



team = np.array([[0,1],[2,3]])

team = {'teste': team}

print(team)
print(type(team))

team = team.tolist()

print(team)
print(type(team))
file_path = 'mydata.json' ## your path variable

json.dump(team, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


# with open('mydata.json', 'w') as f:
#     json.dump(team, f)
    
    
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)

print('----------------')
print(b_new)
print(type(b_new))
print(a_new)
print(type(a_new))
    
# f = open('mydata.json')
# team = json.load(f)

# print(team['tux'])
# print(team['tux']['health'])
# print(team['tux']['level'])

# print(team['beastie'])
# print(team['beastie']['health'])
# print(team['beastie']['level'])

# # when finished, close the file
# f.close()

# for i in team.values():
#     print(i)    


#%%

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

numpyArrayOne = numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])

# Serialization
numpyData = {"array": numpyArrayOne}
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
print("Printing JSON serialized NumPy array")
print(encodedNumpyData)

# Deserialization
print("Decode JSON serialized NumPy array")
decodedArrays = json.loads(encodedNumpyData)

finalNumpyArray = numpy.asarray(decodedArrays["array"])
print("NumPy Array")
print(finalNumpyArray)