# Costa Rica Institute of Technology
# Course: MP-6122 Pattern Recognition
# Student: Jose Martinez Hdez
# Year: 2022
# Laboratory 3: CNN and ANN - MNIST Classification in Keras using a Jetson Nano

import cv2
import tflearn
from image_processor import  ImageProcessor
from tensorflow.python.framework import ops

# name of the opencv window
cv_window_name = "MNIST Camera"
CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    ops.reset_default_graph()
    
    # Include the input layer, hidden layer(s), and set how you want to train the model
    #Inputs
    net = tflearn.input_data([None, 784])
    
    #Hidden layers
    net = tflearn.fully_connected(net, 100, activation = 'ReLU')
    
    #Output
    net = tflearn.fully_connected(net, 10, activation = 'softmax')
    
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    
    # This model assumes that your network is named "net"    
    model = tflearn.DNN(net)
    return model

# handles key presses
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global processor
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('w')):
        processor.p1 +=10
        print('processor.p1:' + str(processor.p1))
    elif (ascii_code == ord('s')):
        processor.p1 -=10
        print('processor.p1:' + str(processor.p1))
    elif (ascii_code == ord('a')):
        processor.p2 +=10
        print('processor.p2:' + str(processor.p2))
    elif (ascii_code == ord('d')):
        processor.p2 -=10
        print('processor.p1:' + str(processor.p2))
    return True

def main():
    # Reload a fresh Keras model from the saved model:
    model = build_model()
    model.load('saved_model/my_model')

    print("Model loaded!")

    # Test image
    processor = ImageProcessor()
    # input_image = cv2.imread(test_image)
    # cropped_input = processor.preprocess_image(input_image)
    
    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

    if ((cap == None) or (not cap.isOpened())):
        print ('Could not open camera.  Make sure it is plugged in.')
        # print ('file name:' + input_video_file)
        print ('Also, if you installed python opencv via pip or pip3 you')
        print ('need to uninstall it and install from source with -D WITH_V4L=ON')
        print ('Use the provided script: install-opencv-from_source.sh')
        exit_app = True
        exit()
    exit_app = False
    while(True):
        ret, input_image = cap.read()

        if (not ret):
            print("No image from from video device, exiting")
            break

        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            exit_app = True
            break
        cropped_input, cropped = processor.preprocess_image(input_image)
        output = model.predict(cropped_input.reshape(1, 28, 28, 1))[0]
        predict_label = output.argmax()
        percentage = int(output[predict_label] * 100)
        label_text = str(predict_label) + " (" + str(percentage) + "%)"
        print('Predicted:',label_text)
        processor.postprocess_image(input_image, percentage, label_text)
        cv2.imshow(cv_window_name, input_image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                exit_app = True
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()