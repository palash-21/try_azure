import numpy as np
import pandas as pd
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import librosa
from keras.models import load_model
from pydub import AudioSegment

def extract_features_audio(data,sample_rate):
	result = np.array([])
	zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
	result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
	stft = np.abs(librosa.stft(data))
	chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
	mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
	result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
	rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
	result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
	mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
	result = np.hstack((result, mel)) # stacking horizontally

	return result

def emotion_decoder(prediction):
    list_pred = prediction.tolist()
    emotion_list = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    emotion_dict = dict(zip(emotion_list,list_pred[0]))
    pred_emotion = ''
    for emot,pred_val in emotion_dict.items():
        max_pred = max(list_pred[0])
        if pred_val==max_pred:
            pred_emotion = emot
    return pred_emotion

modelfile = 'model/model_1.h5'    

model = load_model(modelfile)

UPLOAD_FOLDER = os.getcwd() + '\\uploads'
ALLOWED_EXTENSIONS = {'mp3','wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('get_emotion'))
    return render_template('index.html')

def audio_converter(upload_file_path):
    file_name,file_extension = os.path.splitext(upload_file_path)
    if file_extension.lower() == ".mp3":
        sound = AudioSegment.from_mp3(upload_file_path)
        os.remove(upload_file_path)
        new_file_path = file_name + ".wav"
        sound.export(new_file_path, format="wav")
    else :
        new_file_path = upload_file_path
    
    return new_file_path
    


def predict_emotion():
    for filename in os.listdir(UPLOAD_FOLDER):
        audio_file = filename
        audio_file_path = os.path.join(UPLOAD_FOLDER ,filename)
        break
    
    audio_file_path = audio_converter(audio_file_path)
    data, sample_rate = librosa.load(audio_file_path, duration=2.5, offset=0.6)
    fea = np.array(extract_features_audio(data,sample_rate))
    fea = fea.reshape((-1,162))
    fea = pd.DataFrame(fea)
    fea.reset_index(inplace=True)
    fea = np.array(fea)
    fea = np.expand_dims(fea, axis=2)
    pred = model.predict(fea)
    predicted_emotion = emotion_decoder(pred)
    os.remove(audio_file_path)
    return predicted_emotion


@app.route('/uploads')
def get_emotion():
    
    emotion = predict_emotion()
    return render_template('emotion_recognition.html', variable=emotion)

app.add_url_rule(
    "/uploads", endpoint="get_emotion"
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)