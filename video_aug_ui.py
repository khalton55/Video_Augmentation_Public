#Lib Imports
import cv2
import gradio as gr
import time
import threading
import queue
import requests
import PIL.Image
from io import BytesIO
import os
import json

#File Imports
from utils.util import checkTextInput, checkVideoInput, warningStream

#Globals
producer_thread = None
choice = 'Webcam'
operation = 'None'
should_stop = False
stream_started = False
frame_queue = queue.Queue(maxsize=5)
queue_lock = threading.Lock()

def process_frame(operation, frame, text):
    match operation:
        case 'None':
            return frame
        case 'Grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame
        case 'Text Overlay':
            # Get dimensions of the image
            height, width = frame.shape[:2]  

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)  # Green text color

            # Get the size of the text
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate the bottom-right corner position for the text
            text_x = width - text_width - 10  # 10 pixels padding from the right edge
            text_y = height - 10  # 10 pixels padding from the bottom edge

            # Put the text on the image
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
            return frame
        case _:
            return frame

def streamCapture(input, textInput, event):
    global stream_started, should_stop, operation

    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    #This takes absolutely forever to timeout if not connecting to stream
    #I tried to set my own custom timeout but still WIP
    #TODO add custom timeout
    video_capture = cv2.VideoCapture(input)

    # Full list of Video Capture Properties for OpenCV: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html

    # Select frame size, FPS:
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 10)
    
    if video_capture.isOpened():
        should_stop = False
        stream_started = True
        first_iteration = True
        while not should_stop:
            ret_val, frame = video_capture.read()

            # If frame is read correctly, continue
            if not ret_val:
                break

            # Convert Frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Proccess Frame with operation
            if operation != 'Llava Query':
                frame = process_frame(operation, frame, textInput)

            # frame = PIL.Image.fromarray(frame)

            #Wait
            cv2.waitKey(1)

            with queue_lock:
                if not frame_queue.full():
                    frame_queue.put(frame)

            if first_iteration:
                event.set()
                first_iteration = False

        # Once Video file is done. release 
        video_capture.release()
    else:
        warningStream(choice)
        return

def frame_generator():
    global should_stop, queue_lock, frame_queue
    while not should_stop:
        with queue_lock:
            if not frame_queue.empty():
                frame = frame_queue.get()
                yield frame

def stream_video(filePath, rtspStream, text):
    global stream_started, choice, operation, producer_thread

    if stream_started:
        return

    videoInput = None
    match choice:
        case 'Webcam':
            videoInput = checkVideoInput(choice, None)
        case 'File':
            videoInput = checkVideoInput(choice, filePath)
        case 'RTSP Stream':
            videoInput = checkVideoInput(choice, rtspStream)
        case _:
            videoInput = checkVideoInput(choice, None)
            
    if videoInput == None:
        return

    textInput = None
   
    if operation == 'Text Overlay':
        textInput = checkTextInput(operation, text)

        if textInput == None:
            return

    # ///////////////////////////
    event = threading.Event()

    producer_thread = threading.Thread(target=streamCapture, args=[videoInput, textInput, event])
    producer_thread.start()
    
    #Wait for capture to start
    event.wait()

    for frame in frame_generator():
        #Need to sleep in order to give time for gradio to render
        time.sleep(1/60)
        yield frame

def submit_query(query):
    global stream_started, operation, frame_queue, queue_lock

    #Just in case
    if operation != 'Llava Query':
        return None, None

    if not stream_started:
        gr.Warning('No Stream is Running!')
        return None, None
    
    queryText = None
    queryText = checkTextInput(operation, query)

    if queryText == None:
        return None, None
    
    #This will result in the loss of 1 frame to the video stream but oh well
    while True:
        with queue_lock:
            if not frame_queue.empty():
                frame = frame_queue.get()
                break
    
    serverFrame = PIL.Image.fromarray(frame)
    buffer = BytesIO()
    serverFrame.save(buffer, format='JPEG')
    buffer.seek(0)

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, './config/config.json')
    with open(config_path) as f:
        # ASSIGN LLAVA SERVER URL IN CONFIG.json
        data = json.load(f)
        url = data['llava_server_url']

    file = {'image': ('frame.jpeg', buffer, 'image/jpeg')}
    text = {'text': queryText}

    try:
        reply = requests.post(url, data=text, files=file)
        try:
            data = reply.json()  
            response = data['reply']
        except ValueError:
            print("Reply is not in JSON format.")
        buffer.close()
        return frame, response
    except requests.exceptions.HTTPError as error:
        gr.Error(f'ERROR: {error}')

def show_components_input(value):
    global choice
    choice = value
    if value == 'File':
        return gr.update(visible=True), gr.update(visible=False)
    elif value == 'RTSP Stream':
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
def show_components_augment(value):
    global operation
    operation = value
    if value == 'Text Overlay':
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif value == 'Llava Query':
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
def stop_stream():
    global stream_started, should_stop, producer_thread, frame_queue, queue_lock
    if not stream_started:
        return

    should_stop = True
    producer_thread.join()
    print(f'Queue Empty?: {frame_queue.empty()}')
    with queue_lock:
        while not frame_queue.empty():
            frame_queue.get_nowait()
    print(f'Queue Empty?: {frame_queue.empty()}')
    stream_started = False
    time.sleep(1)
    return gr.update(value=None)


with gr.Blocks() as app:
    gr.HTML(f'''
            <h1 style='text-align: center;'>Video Augmentation</h1>
            <p style="text-align: center;">Takes Video files, Webcam input, or RTSP stream and augments them.</p>
            ''')
    video_output = gr.Image(width=680, height=480 ,label='Processed Video', interactive=False,)
    dropDownInput = gr.Dropdown(
        choices=['Webcam', 'File', 'RTSP Stream'], label='Video Input', info='Choose what video format to augment'
    )

    fileUpload = gr.File(label='File Upload', visible=False, interactive=True, file_types=['video'])

    rtspStreamInput = gr.Textbox(label='RTSP Stream', placeholder='Type link to RTSP Stream', visible=False, interactive=True)
    
    dropDownInput.change(show_components_input, dropDownInput, [fileUpload, rtspStreamInput])

    operationDropDown = gr.Dropdown(
        choices=['None', 'Grayscale', 'Text Overlay', 'Llava Query'], label='Augmentation Operation', info='Choose how you would like to augment the video'
    )

    textAugmentInput = gr.Textbox(label='Text Overlay', placeholder='Type Text To Overlay', visible=False, interactive=True)

    queryInput = gr.Textbox(label='Llava Query', placeholder='Type Query', visible=False, interactive=True)

    submitQuery = gr.Button(value='Submit Query', visible=False, interactive=True)

    llavaFrame = gr.Image(width=680, height=480 ,label='Query Frame',visible=False, interactive=False)

    llavaResponse = gr.TextArea(label='Llava Response', visible=False, interactive=False)

    operationDropDown.change(show_components_augment, operationDropDown, [textAugmentInput, queryInput, submitQuery, llavaFrame, llavaResponse])

    startStream = gr.Button(value='Start Stream')

    stopStream = gr.Button(value='Stop Stream')

    submitQuery.click(
        fn=submit_query,
        inputs=[queryInput],
        outputs=[llavaFrame, llavaResponse]
    )

    #Adding dynamic inputs depending on user is weird in gradio thus I send everything in
    #I handle which ones to use depending on user choice
    startStream.click(
        fn=stream_video,
        inputs=[fileUpload, rtspStreamInput, textAugmentInput],
        outputs=[video_output]
    )

    stopStream.click(
        fn=stop_stream,
        outputs=[video_output]
    )


    app.launch() #server_name='0.0.0.0' is you want open to your network

    # rtsp://admin:admin123@97.128.15.18:8554