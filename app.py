import os
import numpy as np
import pickle
from flask import Flask, request, render_template_string, send_from_directory
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from werkzeug.utils import secure_filename

# Load the captioning model and tokenizer
caption_model2 = load_model('caption_model.keras')
with open('tokenizer2.pkl', 'rb') as f:
    tokenizer2 = pickle.load(f)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Image Captioning</title>
  <style>
    body {
      background: #f7f9fc;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      color: #333;
    }
    h1 {
      margin-top: 40px;
      font-weight: 700;
      color: #004d99;
    }
    form {
      background: white;
      padding: 20px 30px;
      margin-top: 30px;
      border-radius: 10px;
      box-shadow: 0 6px 15px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 320px;
    }
    input[type="file"] {
      margin-bottom: 20px;
      cursor: pointer;
    }
    input[type="submit"] {
      background: #007bff;
      border: none;
      color: white;
      padding: 10px 25px;
      font-size: 1rem;
      border-radius: 6px;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }
    input[type="submit"]:hover {
      background: #0056b3;
    }
    h2 {
      margin-top: 40px;
      color: #004d99;
    }
    p.caption-text {
      font-size: 1.2rem;
      font-weight: 500;
      margin: 15px 0 30px 0;
      max-width: 400px;
      text-align: center;
    }
    img {
      max-width: 300px;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      margin-bottom: 40px;
    }
  </style>
</head>
<body>
  <h1>Upload an Image to get Caption</h1>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file" required><br>
    <input type="submit" value="Get Caption">
  </form>

  {% if caption %}
    <h2>Caption:</h2>
    <p class="caption-text">{{ caption }}</p>
    <img src="{{ url_for('uploaded_file', filename=filename) }}" alt="Uploaded Image" />
  {% endif %}
</body>
</html>
"""

def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature.reshape((1, 2048)), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return ' '.join(in_text.split()[1:])

def Image_caption(image_path):
    img = load_img(image_path, target_size=(600, 600))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    model_resnet = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')
    photo_feature = model_resnet.predict(img_array, verbose=0)

    caption = generate_caption(caption_model2, tokenizer2, photo_feature, 51)
    return caption

@app.route('/', methods=['GET', 'POST'])
def home():
    caption = None
    filename = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            caption = Image_caption(filepath)
    return render_template_string(HTML_TEMPLATE, caption=caption, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
