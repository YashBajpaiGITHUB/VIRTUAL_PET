import cv2
import mediapipe as mp
import pyautogui
import time
import socket

def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1

    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 1

    return cnt

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1

HOST = '192.168.25.77'
PORT = 1031

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((HOST, PORT))
    print("Connected to Unity server.")
except ConnectionRefusedError:
    print("Failed to connect to Unity server. Make sure the server is running.")
    exit(1)

try:
    while True:
        end_time = time.time()
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)

        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keyPoints)

            if not prev == cnt:
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif (end_time - start_time) > 0.2:
                    sock.sendall(str(cnt).encode('utf-8'))
                    prev = cnt
                    start_init = False

            drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
