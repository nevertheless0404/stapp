# 가위 바위 보 앱

import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import handTracking as ht                 
from PIL import Image
from streamlit_lottie import st_lottie

# 초기화 설정.
img = np.asarray(Image.open('./images/None.png'))
img0 = np.asarray(Image.open('./images/Kawi.png'))
img1 = np.asarray(Image.open('./images/Bawi.png'))
img2 = np.asarray(Image.open('./images/Bo.png'))
labels = {-1:'--', 0:'Kawi', 1:'Bawi', 2:'Bo'}
hand_images = {-1:img, 0:img0, 1:img1, 2:img2}
random.seed(time.time()) 

def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

# Landmarks를 받아서 벡터 사이의 각도를 반환
def getAngles(lms):
    # 0번 landmakr 가로, 세로 좌표
    base = lms[0][1:]
    lms = np.array( [  (x,y) for id, x, y in lms ])
    vectors = lms[1:] - np.array([base]*20)
    # 마디마디를 연결해서 벡터를 만듬 
    # 축이 하나 밖에 없음
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    # 축을 2개가 되도록 구성
    vectors = vectors/norms
    # 길이가 1인 벡터로 정규화 
    cos = np.einsum( 'ij,ij->i', vectors[:-1], vectors[1:] )
    # Degree 변환
    angles = np.arccos(cos)*180/np.pi
    return angles

# 결과를 해석 
def getResult(human, computer):
    if human == 0 and computer == 0:        
        res = '무승부!'
    elif human == 0 and computer == 1:     
        res = '컴퓨터 승!'
    elif human == 0 and computer == 2:      
        res = '인간 승!'    
    elif human == 1 and computer == 0:      
        res = '인간 승!'
    elif human == 1 and computer == 1:      
        res = '무승부!'
    elif human == 1 and computer == 2:      
        res = '컴퓨터 승!'    
    elif human == 2 and computer == 0:     
        res = '컴퓨터 승!'
    elif human == 2 and computer == 1:     
        res = '인간 승!'
    elif human == 2 and computer == 2:   
        res = '무승부!'   
    else:
        res = '판별 불가~'
    
    return res

# 머신러닝 모형 학습
@st.cache_resource
def initML():
    df = pd.read_csv('data_train.csv')
    X = df.drop(columns=['19']).values.astype('float32')   
    Y = df[['19']].values.astype('float32')                

    knn = cv2.ml.KNearest_create()
    knn.train(X, cv2.ml.ROW_SAMPLE, Y)
    return knn

# 이미지 프레임을 처리해 주는 함수
def image_processor(img, knn, detector):
    img = detector.findHands(img)
    lms = detector.getLandmarks(img)                     # 검출된 landmark 좌표점    
    if lms:
        angles = getAngles(lms)
        angles = angles[np.newaxis, :]
        pred = knn.findNearest(angles.astype('float32'), 3) # 검출.
        human = int(pred[0])                                # 사람의 손모양
    else:                       
        human = -1
    return human

# 로고 Lottie 타이틀 
col1, col2 = st.columns([1,2])
with col1:
    lottie = loadJSON('lottie-rock-paper-scissors.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.title('가위 바위 보~')

img_file_buffer = st.camera_input(' ')

''
col3,col4 = st.columns(2)

result = st.empty()

knn = initML()
my_detector = ht.HandDetector() 

if img_file_buffer:
    with col3:
        st.header(':blue[나]')
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        human = image_processor(img_array, knn, my_detector)
        fig = plt.figure()
        plt.imshow(hand_images[human])
        plt.axis('off')
        st.pyplot(fig)
    
    with col4:
        st.header(':red[컴퓨터]')
        computer = random.randrange(3)
        fig = plt.figure()
        plt.imshow(hand_images[computer])
        plt.axis('off')
        st.pyplot(fig)

        result.title(f'{getResult(human, computer)}')