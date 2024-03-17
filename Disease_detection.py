import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import ttk
import customtkinter as ct
import cv2
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import glob
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten ,Dropout ,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import regularizers
import joblib

def crop_image(image,size):
    
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Applying Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding the image to create a binary mask
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    # Performing morphological operations to remove noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Finding contours in the binary mask
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the brain part of the image has the largest contour
    c = max(contours, key=cv2.contourArea)

    # Getting the bounding rectangle of the brain part
    x, y, w, h = cv2.boundingRect(c)

    # Cropping the image around the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    # Resizing cropped image to the needed size
    resized_image = cv2.resize(cropped_image, size)
    
    

    return np.expand_dims(resized_image,axis=0)


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
    str = file_path
    if file_path:
        image = Image.open(file_path)

        image.thumbnail((300,400))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        image_label.place(x=85,y=75,)


def pred(event):
    print(str)

    # Set the path to your saved model file
    model_path1 = 'C:/Users/jatin/Untitled Folder 6/new/vgg_bone_model.h5'
    model_path2 = 'C:/Users/jatin/Untitled Folder 6/class_model.h5'
    model_path3 =  'C:/Users/jatin/Downloads/vgg_model.h5'
    # Load the model
    model_bone = tf.keras.models.load_model(model_path1)
    model_class = tf.keras.models.load_model(model_path2)
    model_tumor = tf.keras.models.load_model(model_path3)

    #Reading image
    image=cv2.imread(str)

    # Cropping iamge
    img=crop_image(image,(128,128))

    threshold = 0.5
    pred_bin = ((model_class.predict(img)) > threshold).astype(int)[0][0]
    
    if bool(pred_bin):
       
        pred = ((model_bone.predict(img)) > threshold).astype(int)[0][0]
        if bool(pred):
           
             str2= '''                   !! Bone Fracture !!
                                     Seek Immediate Medical Attention
                Immobilize the Area:While waiting for medical help, try to immobilize the injured area to minimize
                movement and further pain. You can use a splint, sling, or other supportive material if available,
                but don't attempt to set the bone yourself.
                Reduce Swelling:Apply ice packs wrapped in a towel to the affected area to reduce swelling and pain.
                
                '''
        
        else:
            str2='''         !! No Fracture !!
                    Here Some Suggestions to prevent bone fractures:
                    Calcium and Vitamin D:These are essential nutrients for bone health. Aim for a diet rich 
                    in calcium-containing foods like dairy products, leafy greens, and fortified foods.
                    Protein: Protein is another crucial building block for bones. Include protein sources like 
                    lean meats, poultry, fish, beans, lentils, and nuts in your diet.
                    Fruits and Vegetables: They provide essential vitamins, minerals, and antioxidants that 
                    contribute to overall health and bone strength.

'''
    
    else:
        pred = ((model_bone.predict(img)) > threshold).astype(int)[0][0]
        if bool(pred):
            
                    str2 = '''
    Brain Tumour - +ve  
    Symptoms: 
    headaches, seizures, cognitive
    changes, balance
    problems, vision or hearing
    changes.

    Treatment: Optionsinclude surgery, radiation therapy, chemotherapy, and 
    targeted therapy, depending on tumor type and patient health.

    Diagnosis
    Challenges: Diagnosing brain tumors can be tricky due to varied symptoms and the need
    for specialized imaging tests like MRI or CT scans. '''
       
        else :
            
               str2 = ''' Brain Tumour  -ve
          Seek Immediate Medical Attention: If experiencing severe symptoms like sudden headaches, seizures, call hospital.

          Consult a Healthcare Professional: Schedule an appointment with a doctor or neurologist promptly for evaluation, .

          Follow Medical Advice: include surgery, radiation therapy, or chemotherapy.

         Provide Support: Offer emotional support and practical assistance to the affected individual.

         Monitor Symptoms: Keep track of any changes in symptoms.'''

    
    label = tk.Label(root, text=str2)
    label.pack(side="right",padx=30)

    





def main():
    # Creating root frame
    root = tk.Tk()
    root.title("Medical Diagnostic Application")
    
    str = ""

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])

    str = file_path
    
    heading_label = tk.Label(root, text="Team Technocrats", font=("Helvetica", 18, "bold"), padx=20, pady=10)
    heading_label.pack()
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    window_width = int(screen_width * 0.7)
    window_height = int(screen_height * 0.5)
    
    root.geometry(f"{window_width}x{window_height}")
    
    
    bg_image = Image.open("C:/Users/jatin/Downloads/starline.jpg")
    
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
    
    
    
    
    button_font = ('Arial', 12, 'bold')
    
    
    
    select_button = tk.Button(root, text="Select Image", command=open_image, bg="#6A5ACD", font = "button_font")
    select_button.pack(pady=10 , side="top")
    
    
    pred_button = tk.Button(root, text="Get Impression",  bg="#6A5ACD", font="button_font")
    
    
    pred_button.bind("<Button-1>", pred)
    pred_button.pack(pady=10, side="bottom")
    
    image_label = tk.Label(root)
    image_label.pack(anchor="w")
    
    root.mainloop()
    
