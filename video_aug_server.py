import os
import io
import time
from PIL import Image

#Flask
from flask import Flask, request, jsonify

#NanoLLM
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser


model_name = 'liuhaotian/llava-v1.5-13b'
model = None
chat_history = None
args = None
isStreaming = False
system_prompt = """
        You are LLaVA, a multimodal AI designed to analyze, describe, and interpret visual data.
        When given an image, provide clear, descriptive, and insightful observations. 
        Combine your visual understanding with textual context to answer questions or provide
        assistance. 
        Ensure responses are user-friendly, relevant, and concise.
        Only respond in English!
    """

app = Flask(__name__)

def setup():
    try:
        global model, chat_history, model_name, args, isStreaming, system_prompt
        
        #Setting Args
        parser = ArgParser(extras=ArgParser.Defaults)
        args = parser.parse_args()
        args.vision_api = 'hf'
        args.max_content_len = 100
        args.system_prompt = system_prompt
        args.min_new_tokens = 100
        args.max_new_tokens = 300
        print(args)

        #Load model
        print('Loading LLAVA...')
        model = NanoLLM.from_pretrained(
            model=model_name,
            api='mlc',
            quantization='q4f16_ft',
            max_content_len = args.max_content_len,
            vision_api = args.vision_api,
            vision_model = args.vision_model,
            vision_scaling = args.vision_scaling,
            print_stats=False
            
        )
        print('LLAVA Loaded')


        print(f"DOES THIS MODEL HAVE VISION: {model.has_vision}")
        print(f"VISION API : {args.vision_api} >>  Vision Model : {args.vision_model} >> {args.chat_template}")

        #Create Chat History
        print('Creating Chat History...')
        chat_history = ChatHistory(model, args.chat_template, args.system_prompt)
        chat_history.append(role="user", text="What is machine learning? -explain in english")
        print('Chat History created')

        #Embeddings and Warm up Reply
        print("Making Embeddings and generating warm up reply...")
        embedding, _ = chat_history.embed_chat()
        reply = model.generate(
            embedding,
            kv_cache= chat_history.kv_cache,
            max_new_tokens= args.max_new_tokens,
            min_new_tokens= args.min_new_tokens,
            do_sample= args.do_sample,
            repetition_penalty= args.repetition_penalty,
            temperature= args.temperature,
            top_p= args.top_p,
            streaming= isStreaming
        )

        print(f"Warm up reply: {reply}")

        return 'Setup Completed!'
    except Exception as error:
        print(f'ERROR: {error}')
        return 'Setup failed - See Error Above'
    
#Flask Route: /query POST
@app.route('/query', methods=['POST'])
def query():
    global model, chat_history, args, isStreaming, system_prompt

    #Check for user input
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Image and text are required.'}), 400
    
    #Gets Image and Text data from request
    image_file = request.files['image']
    text_data = request.form['text']
	
    #Get prompt
    prompt = text_data.strip()
    
    print(f'Prompt: {prompt}')

    #Sanitize query input
    if len(prompt) == 0:
        return jsonify({'reply': 'Query cannot be empty!'}), 400
    
    #See if image is in right format
    image = None
    try:
        image = Image.open(image_file)
    except Exception as e:
        return jsonify({'error': 'Invalid image format.', 'details': str(e)}), 400
    
    #Reset Chat History
    chat_history.reset(system_prompt=system_prompt)
    
    #Add image
    chat_history.append('user', image)

    #Add user prompt 
    chat_history.append('user', prompt)

    #Generate chat embeddings
    embedding, _= chat_history.embed_chat()

    #Generate chat bot reply
    try:
        reply = model.generate(
            embedding, 
            kv_cache=chat_history.kv_cache,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
            streaming=isStreaming
        )

        response = reply
        response = response.replace("\n", "").replace("</s>", "").replace("<s>", "")

        print(f'Response: {response}')

        return jsonify({'reply': f'{response}'}), 200
    except Exception as error:
        print(f'ERROR: {error}')


#Runs setup and starts app
if __name__ == '__main__':
    print(setup())
    app.run(host='0.0.0.0', debug=True, port=5000)
