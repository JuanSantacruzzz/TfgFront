from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from os import listdir
from aplicacion.forms import UploadForm
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
Bootstrap(app)	


import tensorflow.keras as keras
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from cv2 import cv2

@app.route('/', methods=['get', 'post'])
def inicio():
	
	lista=[]
	for file in listdir(app.root_path+"/static/img"):
		lista.append(file)
	return render_template("inicio.html",lista=lista)



@app.route('/upload', methods=['get', 'post'])
def upload():
	form= UploadForm() # carga request.from y request.file
	if form.validate_on_submit():
		f = form.photo.data
		filename = secure_filename(f.filename)
		f.save(app.root_path+"/static/imagenesProcesadas/"+filename)
		return redirect(url_for('resultado',filename=filename))
	return render_template('upload.html', form=form)	


@app.route('/resultado/<filename>', methods=['get','post'])
def resultado(filename=None):

	new_model = keras.models.load_model(app.root_path+'/content/path_to_my_model.h5')
	img_path=app.root_path+'/static/imagenesProcesadas/'+filename
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = new_model.predict(x)
	print('Predicted:', preds)

	# Mapa de color
	african_elephant_output = new_model.output[:, 1]
	last_conv_layer = new_model.get_layer('block5_conv4')

	disable_eager_execution()

	a = tf.constant(1)
	b = tf.constant(2)
	c = a + b
	
	grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
	pooled_grads = K.mean(grads, axis=(0, 1, 2))
	iterate = K.function([new_model.input], [pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])
	for i in range(512):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap) 

	print(img_path)
	# We use cv2 to load the original image
	img = cv2.imread(img_path)
	
	# We resize the heatmap to have the same size as the original image
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

	# We convert the heatmap to RGB
	heatmap = np.uint8(255 * heatmap)

	# We apply the heatmap to the original image
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	# 0.4 here is a heatmap intensity factor
	superimposed_img = heatmap * 0.4 + img

	print("antes")
	print(app.root_path+'/static/mapaCalor/'+filename)
	# Save the image to disk
	cv2.imwrite(app.root_path+'/static/mapaCalor/'+filename, superimposed_img)

	return render_template("resultado.html",filename=filename,preds=preds)

@app.errorhandler(404)
def page_not_found(error):
	return render_template("error.html",error="Página no encontrada..."), 404

