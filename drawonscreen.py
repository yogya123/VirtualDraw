import cv2
import mediapipe as mp
import numpy as np


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):   
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

coordinates = []

def det_hand(image):

  original=image.copy()
  global coordinates 
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands
 
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
      return None
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      point_x= int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
      point_y= int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
      current = point_x, point_y
      coordinates.append(current)
      for i in range (len(coordinates)):
          original = cv2.circle(original, (coordinates[i][0], coordinates[i][1]), radius=8, color=(200, 10, 16), thickness=-1)
    print(point_y, point_x)
    return original

  
camera = cv2.VideoCapture(0)
output_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (1280, 960))

while True:
    x,y=(0,0)
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = cv2.flip(frame, 1)
    frame = apply_brightness_contrast(frame, 20, -2)
    final_image= det_hand(frame)
    if final_image is not None:
        cv2.resize(final_image, (1280, 960))
        # final_image = cv2.rectangle(final_image, (500, 10), (700,90), (0,0,0), -1)
        # final_image = cv2.putText(final_image, "Eraser", (500,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        
        cv2.imshow('please', final_image)
        output_writer.write(cv2.resize(final_image, (1280,960)))


    if cv2.waitKey(1)==ord('q'):
        break

