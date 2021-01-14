from flask import Flask, render_template
import gen_image_func 
import os

app = Flask(__name__)

@app.route('/')
def index():
	os.system('python3 gen_image.py')
	filename = 'http://127.0.0.1:5000/static/images/img.jpg'
	
	return render_template('index.html', filename=filename)


