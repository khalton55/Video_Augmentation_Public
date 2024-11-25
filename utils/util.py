import gradio as gr
import json
import os



def checkString(string, varName):
    if not isinstance(string, str):
        gr.Warning(f'{varName} must be a string!')
        return None
    string = string.strip()
    if len(string) == 0:
        gr.Warning(f'{varName} cannot be an empty string!')
        return None
    else:
        return string
    
def checkFile(file):
    if not file:
        gr.Warning('No file selected!')
        return None
    else:
        return file

#Displays Gradio warnings depending on input type when attempting to stream
def warningStream(choice):
    match choice:
        case 'Webcam':
            gr.Warning("Unable to open camera")
        case 'RTSP Stream':
            gr.Warning('Unable to open RTSP Stream')
        case 'File':
            gr.Warning('Unable to open file')
        case _:
            gr.Warning('Unmatched selection! (Shouldnt happen :( )')
    return

def checkVideoInput(choice, input):
    match choice:
        case 'Webcam':
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, '../config/config.json')
            with open(config_path) as f:
                # ASSIGN CAMERA ADDRESS IN CONFIG.json
                # "/dev/video0"
                data = json.load(f)
                camera_address = data['camera_address']
            
                return camera_address
        case 'File':
            file = checkFile(input)
            return file
        case 'RTSP Stream':
            stream = checkString(input, 'RTSP Stream')
            return stream
        case _:
            gr.Warning('Unmatched Selection!')
            return None
        
def checkTextInput(operation, text):
    match operation:
        case 'Text Overlay':
            text = checkString(text, 'Text Overlay')
            return text
        #Repeat Code but placeholder for different query check
        case 'Llava Query':
            text = checkString(text, 'Llava Query')
            return text
        case _:
            gr.Warning('Unmatched Selection!')
            return None
        