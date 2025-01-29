# 모듈을 사용하는 Hand Tracking

# Hand Tracking 최소 코드

import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionConf=0.5, trackConf=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils        
        self.mpHands =  mp.solutions.hands    
        # hands 객체 생성          
        self.hands = self.mpHands.Hands(              
            static_image_mode=self.mode,               
            max_num_hands=self.maxHands,               
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf)   

    def findHands(self, imgRGB, flag=True):
        self.res = self.hands.process(imgRGB)                   # RGB 이미지를 받아서 처리

        if(self.res.multi_hand_landmarks):                      # 손이 검출되었다면
            for a_hand in self.res.multi_hand_landmarks:        # 하나하나의 손을 가져온다
                if flag:
                    self.mpDraw.draw_landmarks(imgRGB, a_hand, self.mpHands.HAND_CONNECTIONS)  
        return imgRGB
    
    def getLandmarks(self, img, handNo=0):
        lms = []                                             # 반환될 landmark 의 리스트
        if(self.res.multi_hand_landmarks):                   # 손이 검출되었다면
            a_hand = self.res.multi_hand_landmarks[handNo]   # 특정 손을 가져온다
            for id, lm in enumerate(a_hand.landmark):        # 0~20까지 하나씩 처리
                h, w, c = img.shape                          # shape속성에서 이미지의 크기 픽셀 단위로 추출
                cx, cy = int(lm.x * w), int( lm.y * h)       # 실수 비율을 실제 픽셀 포지션으로 변환    
                lms.append([id,cx,cy])             
        return lms   


def main():
    my_cap = cv2.VideoCapture(0)                 
    my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     
    my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)   
    my_detector = HandDetector()                  
    while True:
        _, img = my_cap.read()               
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
        imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화
        lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점        

        # 검출이 성공한 경우만 출력.
        if lms:
            print(lms)

        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  
        cv2.imshow("Image", imgBGR)
        if cv2.waitKey(1) & 0xFF == ord('q'):            
            break

if __name__ == "__main__":
    main()
