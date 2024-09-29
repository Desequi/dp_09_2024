import cv2
import pandas as pd
import numpy as np
from imagededup.methods import CNN

def create_matrix(rows, cols, value=0.):
    return [[value for _ in range(cols)] for _ in range(rows)]

def get_key_frame(name, key):
    cap = cv2.VideoCapture(name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, key)
    ret, frame = cap.read()
    return frame


def key_frames(name, keys):
    cap = cv2.VideoCapture(name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

def key_dedup(name):
    cnn = CNN()
    encodings = cnn.encode_images(name)
    return encodings

def key_dedup_one(name):
    cnn = CNN()
    encodings = cnn.encode_image(image_array=name)
    return encodings

def open_db():
    df = pd.read_csv('db.csv')
    return df

def read_specific_frame_cnn(video_path, frame_numbers):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count_frame = 0
    det = ' '

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        name = video_path[:-4]+str(frame_number)
        cnn = CNN()
        encodings = cnn.encode_image(image_array=frame)
        enc  = ['%.5f' % encodings[0][elem] for elem in range(0,len(encodings[0]),4)]
        str_ = name + det
        for elem in enc:
            str_ = str_ + elem + det

        f = open('out.csv', 'a')
        f.write(str_ + '\n')
        f.close()
def save_specific_frame(video_path, frame_numbers, key):
    cap = cv2.VideoCapture(video_path)
    det = ' '

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        res, frame = cap.read()
        cv2.imwrite('D:\\img\\' +key+'_'+str(frame_number)+'.png', frame)
    # return frames