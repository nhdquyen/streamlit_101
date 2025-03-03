import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

st.header('HUMBLE MACHINE LEARNING MODEL')

menu = ['Model Info', 'Testing']
choice = st.sidebar.radio('Menu', menu)
if choice == 'Model Info':
    st.image('media/Should-I-Choose-Machine-Learning-or-Big-Data.jpeg')
    st.text('This is Quyên. Thank you for testing with my very first Machine Learning model.')
    name = st.text_input('Before going any further, may I know your name?')
    if len(name) > 0:
        st.write(f'Hello, {name}. A nice name you have!')
        st.text('This model is compiled with two main parts: 1-Feature Extraction which is the pretrained \nDenseNet121 and 2-Classification with one Global Average layer and finally Softmax on top.')
        st.text('DenseNet121 was chosen for its light size and ability to capture feature importances which \nis a good fit for classifying VND banknotes.')
        st.text('At feature extraction layers constructed with 427 layers, or base model for short, is \nlocked with 320 pretrained layers with imagenet.')
        
        st.write("Click [here](https://arxiv.org/pdf/1608.06993.pdf) for further study. Now, have fun testing out!")

# elif choice == 'Read Data':
#     df = pd.read_csv('media/AB_NYC_2019.csv')
#     st.dataframe(df)

elif choice == 'Testing':
    model_path = 'model/WP8_vn_banknote_model.h5'
    model = tf.keras.models.load_model(model_path)
    class_names = ['1,000', '10,000', '100,000', '2,000', '20,000', '200,000', '5,000', '50,000', '500,000']

    col1, col2 = st.beta_columns(2)

    with col1: #upload
        fileup = st.file_uploader('UPLOAD FILE', type = ['jpg', 'png', 'jpeg'])
        if fileup != None:  
            st.image(fileup)   
            image_np = np.asarray(bytearray(fileup.read()), dtype = np.uint8)    
            img = cv2.imdecode(image_np,1)
            img = cv2.resize(img, (224,224))
 
            img_array  = np.expand_dims(img, axis=0)
            prediction = model.predict(img_array)
            if prediction[0].max() < 0.45:
                st.write('You are funny!')
            else:
                pred_indices = np.argmax(prediction, axis = 1)
                st.write(class_names[pred_indices[0]], 'vnd')

    with col2: #webcam 
        cap = cv2.VideoCapture(0)  # device 0
        run = st.checkbox('Show Webcam')
        capture_button = st.checkbox('Capture')

        captured_image = np.array(None)


        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        FRAME_WINDOW = st.image([])
        while run:
            ret, frame = cap.read()        
            # Display Webcam
            #frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
            FRAME_WINDOW.image(frame)

            if capture_button: # to capture
                captured_image = frame
                break

        cap.release()

        if  captured_image.all() != None:
            #st.image(captured_image)
            st.write('Image is capture:')

            #Resize the Image according with your model
            captured_image = cv2.resize(captured_image, (224,224))
            #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
            img_array  = np.expand_dims(captured_image, axis=0)
            #Check the img_array here
            prediction = model.predict(img_array)
            if prediction[0].max() < 0.45:
                st.write('You are funny!')
            else:
                pred_indices = np.argmax(prediction, axis = 1)
                st.write(class_names[pred_indices[0]], 'vnd')
            
            