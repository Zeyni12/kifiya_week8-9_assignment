from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
       try:
       except ValueError:    
    return render_template('predict.html')       

if __name__ == '__main__':
    model.run()