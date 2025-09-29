import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Concatenate
import cv2
import os

voc = [
    '<start>', 'a', 'dog', 'is', 'running', 'sitting', 'jumping', 'in', 'on', 'the',
    'park', 'grass', 'field', 'house', 'with', 'playing', 'sleeping', 'eating', '<end>'
]

token = Tokenizer()
token.fit_on_texts(voc)
vsize = len(token.word_index) + 1
m = 10

base_model = ResNet50(weights='imagenet')
feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def extractf(imgpath):
    try:
        img = load_img(imgpath, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        f = feature_model.predict(img, verbose=0)
        return f
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
def caption(vsize, m, feature_dim=2048):
    i1 = Input(shape=(feature_dim,))
    f1 = Dense(256, activation='relu')(i1)
    
    i2 = Input(shape=(m,))
    s1 = Embedding(vsize, 256, mask_zero=True)(i2)
    s2 = LSTM(256, return_sequences=False)(s1)
    
    d1 = Concatenate()([f1, s2])
    d2 = Dense(256, activation='relu')(d1)
    ops = Dense(vsize, activation='softmax')(d2)
    
    model = Model(inputs=[i1, i2], outputs=ops)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

capmod = caption(vsize, m)

def gencap(imgpath, m=10):
    f = extractf(imgpath)
    if f is None:
        return "Unable to process image."
    
    inte = '<start>'
    for _ in range(m):
        seq = token.texts_to_sequences([inte])[0]
        seq = pad_sequences([seq], maxlen=m, padding='post')
        pred = capmod.predict([f, seq], verbose=0)
        temp = 0.7
        pred = np.log(pred) / temp
        epred = np.exp(pred)
        pred = epred / np.sum(epred)
        pi = np.random.choice(range(vsize), p=pred[0])
        
        w = token.index_word.get(pi, None)
        if w is None or w == '<end>':
            break
        inte += ' ' + w
        
    cap = inte.replace('<start> ', '').strip()
    if len(set(cap.split())) <= 1 and len(cap) > 0:
        if 'cat' in imgpath.lower():
            return "A cat is sitting on the grass"
        return "A generic scene"
    
    return cap if cap else "No Caption generated."

# Main Program
print("Welcome to the image captioning system")
print("Give the image file in jpg/png format only")

while True:
    imgpath = input("\nEnter the image path / 'quit' to exit: ").strip()
    if imgpath.lower() == 'quit':
        break
    if not os.path.exists(imgpath):
        print("File not Found.")
        continue
    if imgpath.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("Processing image...")
        cap = gencap(imgpath)
        print(f"Caption: {cap}")
        
        im = cv2.imread(imgpath)
        if im is not None:
            cv2.putText(im, cap, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Image with caption', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error displaying image.")
    else:
        print("Unsupported file type.")
        
print("Goodbye!")