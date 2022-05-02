import flask
import flask_cors
import dlib
import face_recognition
import face_recognition_models
import joblib
import matplotlib
import numpy
import skimage
import sklearn
import scipy
import imageio
import PIL
import base64
import io

from emotion_classifier import EmotionClassifier

PRODUCTION = True

# PRODUCTION = True

# < ~/Desktop/azure-docker >

# sudo docker build -t emotionclassifier.azurecr.io/classifier-docker:latest .

# sudo docker image ls

# sudo docker run emotionclassifier.azurecr.io/classifier-docker

# stop azure app

# sudo docker push emotionclassifier.azurecr.io/classifier-docker:latest

# check container has updated with recent push

# start azure app while viewing stream log


app = flask.Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = flask_cors.CORS(app)

model = joblib.load('models/MLP1-os')

clf = EmotionClassifier(model)

__test__data = face_recognition.load_image_file('benshapiro.jpg')

@app.route('/')
def root():
    return "200"

@app.route('/info')
def versions():
    s = 'This server is running flask version: {}<br/>'.format(flask.__version__)
    s += 'This server is running flask_cors version: {}<br/>'.format(flask_cors.__version__)
    s += 'This server is running dlib version: {}<br/>'.format(dlib.__version__)
    s += 'This server is running face_recognition version: {}<br/>'.format(face_recognition.__version__)
    s += 'This server is running face_recognition_models version: {}<br/>'.format(face_recognition_models.__version__)
    s += 'This server is running joblib version: {}<br/>'.format(joblib.__version__)
    s += 'This server is running matplotlib version: {}<br/>'.format(matplotlib.__version__)
    s += 'This server is running numpy version: {}<br/>'.format(numpy.__version__)
    s += 'This server is running skimage version: {}<br/>'.format(skimage.__version__)
    s += 'This server is running sklearn version: {}<br/>'.format(sklearn.__version__)
    s += 'This server is running scipy version: {}<br/>'.format(scipy.__version__)
    s += 'This server is running imageio version: {}<br/>'.format(imageio.__version__)
    s += 'This server is running PIL (Pillow) version: {}<br/>'.format(PIL.__version__)
    return s

@app.route('/testpredict', methods=['GET'])
@flask_cors.cross_origin()
def test_predict():
    try:
        pred = clf.predict_from_data(__test__data)
        resp = flask.jsonify({'msg':str(pred)})
        return resp, 200
    except:
        return {'msg':'Error predicting image'}, 500

@app.route('/predict', methods=['POST'])
@flask_cors.cross_origin()
def predict():
    try:
        data = flask.request.data
        b64 = base64.b64decode(data)
        buf = io.BytesIO(b64)
        loaded = face_recognition.load_image_file(buf)
        pred = clf.predict_from_data(loaded)
        resp = flask.jsonify({'msg':str(pred)})
        return resp, 200
    except:
        return {'msg':'Cannot predict image'}, 500

if __name__ == '__main__':
    if PRODUCTION:
        app.run(debug=True,host='0.0.0.0',port=8080) 
    else:
        app.run()   