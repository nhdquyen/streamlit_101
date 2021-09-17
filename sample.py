# Sample code for Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

PATH = 'media/AB_NYC_2019.csv'

menu = ['Home', 'Read Data', 'Display Images', 
        'Play Videos', 'Show Webcam', 'Play Music',
        'About me']

choice = st.sidebar.selectbox('What puppy can do?', menu)

if choice=='Home':
    st.title("Puppy Wonderland")
    st.header("My first app")

    st.write("")
    st.write("My puppy can do anything!")

    st.image('media/isle_of_dog.gif',
            caption="My lovely black puppy",
            use_column_width='auto')

    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    col1, col2, col3 = st.columns(3)

    # NAME
    with col1:
        name = st.text_input('Enter your puppy name:')
        if name!="":
            st.write(name, 'is a cute name!')

    # AGE
    with col2:
        age = st.slider('Choose your puppy age', min_value=1, max_value=20)
        st.write('Your puppy is', age, 'years old!')

    # FOOD
    with col3:
        food = st.multiselect('What does your puppy love to eat?', ['Bone', 'Sausage', 'Caviar'])
        if food==['Bone']:
            st.write('He needs to bark first!')
        elif food==['Sausage']:
            st.write("Nah, it's expensive but... okay!")
        else:
            st.write('Are you sure?')

elif choice=='Read Data':
    # Cache the function output
    @st.cache()
    def load_data(path):
        return pd.read_csv(path)
    
    st.title('Hot Dog Summer!')
    st.image('media/dog-beach-lifesaver.png')

    df = load_data(PATH) 
    st.dataframe(df)

    figure, ax = plt.subplots() # A must in Streamlit
    df.groupby('neighbourhood_group')['price'].mean().plot(kind="barh", ax=ax)
    st.pyplot(figure)
    st.write('This is a cool chart!')

    price = st.slider('Consider about the price', min_value=10, max_value=100)
    filter = df[df['price']<price]
    st.map(filter[['latitude', 'longitude']])

elif choice=='Display Images':
    st.title('Puppy can display images!')
    photo_uploaded = st.file_uploader('Choose your best puppy photo', ['png', 'jpg', 'jpeg'])
    if photo_uploaded != None:
        image_np = np.asarray(bytearray(photo_uploaded.read()), dtype=np.uint8)
        # print(image_np)
        # print(image_np.shape)
        img = cv2.imdecode(image_np, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)

        st.write(photo_uploaded.size)
        st.write(photo_uploaded.type)

elif choice=='Play Videos':
    st.title('Puppy can play videos!')
    st.warning("Sounds available on local computer ONLY")
    video_uploaded = st.file_uploader('Import your amazing video', type=['mp4'])
    if video_uploaded != None:
        st.video(video_uploaded)

elif choice=="Show Webcam":
    st.title("Webcam Live Feed!")
    st.warning("Work on local computer ONLY")
    run = st.checkbox('Show!')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write("Stop!")

elif choice=='Play Music':
    st.title("Puppy can rock!")
    st.warning("Sounds available on local computer ONLY")
    audio_uploaded = st.file_uploader("Choose your fav song")
    if audio_uploaded != None:
        audio = audio_uploaded.read()
        st.audio(audio, format="audio/mp3")

elif choice=='About me':
    st.success('An awesome guy!')
    st.image('media/9e1b49d166612f7a7846aa5b77b871c7.gif')
    st.balloons()
