# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:43:53 2020

@author: Sergio
"""


import requests
import json

url = 'http://192.168.15.12:5000/compare'

data = ['str1', 'str2']
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)

