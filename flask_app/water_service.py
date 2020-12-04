# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:07:11 2020

@author: Sergio
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:23:41 2020

@author: Sergio
"""


from flask import Flask, request, redirect, url_for, flash, jsonify

import os

app = Flask(__name__)



@app.route('/')
@app.route('/index')
def hello_world():
    return '''<!DOCTYPE html>
                <html>
                <body>
                
                <h1>HOLLY SHIT IT WORKS</h1>
                
                <img src="/static/coco.jpg" alt="Girl in a jacket">
                
                <h2>APPROVED</h2>
                
                </body>
                </html>
                '''


    
@app.route('/api', methods=['POST'])
def makecalc():
    data = request.get_json()
    # prediction = np.array2string(model.predict(data))

    print('oi')
    return jsonify(data)
   

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    print(data[0], data[1])
    
    os.system("python comparison_v1.py " + data[0] + " " + data[1])  
    
    return 'Files Compared', 200


    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
