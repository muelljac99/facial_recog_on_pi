import io
import os
import time
import picamera
import picamera.array
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow import keras


def img_capture():
    with picamera.PiCamera() as camera:
        camera.start_preview()
        time.sleep(2)
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            image = stream.array
    image = image[:, :, ::-1]
    return image

def detect_and_crop(mtcnn, image):
    faces = mtcnn.detect_faces(image)
    box = faces[0]['box']
    w = int(box[2]+.2*box[2])
    h = int(box[3]+.2*box[3])
    x1 = int(box[0]-.1*box[2])
    y1 = int(box[1]-.1*box[3])
    cropped_image = image[y1:y1+h, x1:x1+w, :] 
    
    return cropped_image, (x1, y1, w, h)

def show_bounding_box(image, bounding_box):
    x1, y1, w, h = bounding_box
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()
    return
        
def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def run_model(model, face):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    face = face.reshape(1, 160, 160, 3)
    
    model.set_tensor(input_details[0]['index'], face)
    model.invoke()
    
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

def read_image(path):
    image = cv2.imread(path)[:, :, ::-1]
    return image
        
mtcnn = MTCNN()
image1 = img_capture()
#image1 = read_image('/home/pi/Desktop/Lab3/myECE.jpg')
cropped_img1, box1 = detect_and_crop(mtcnn, image1)
#show_bounding_box(image1, box1)

os.chdir(r'/home/pi/Desktop/Lab3')
cv2.imwrite('input.jpg', cv2.cvtColor(cropped_img1, cv2.COLOR_RGB2BGR))

#interpreter = tf.lite.Interpreter('/home/pi/Desktop/inception_model.tflite')
#interpreter.allocate_tensors()
#tf_input = pre_process(cropped_img1)
#output1 = run_model(interpreter, tf_input)