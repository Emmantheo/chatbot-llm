import os
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPEN_AI_KEY"))

from flask import Flask, render_template, request, jsonify, session
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.memory import ChatMemoryBuffer
from flask_session import Session
from flask_swagger_ui import get_swaggerui_blueprint
import json
from datetime import datetime
import secrets


# Loading environment variables from .env file
load_dotenv()

# Accessing environment variables
api_key = os.getenv('OPEN_AI_KEY')
print("API Key:", api_key)

app = Flask(__name__)

# Secret key for session management
app.secret_key = secrets.token_urlsafe(16)
print(app.secret_key)

SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "NBS Chat"
    },)
app.register_blueprint(swaggerui_blueprint)

# Accessing environment variables
host_no = os.getenv("host")
port_no = os.getenv("port")
debug_mode = os.getenv("DEBUG") 

# Setting up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Loading the index
PERSIST_DIR = "./store"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Setting up OpenAI language model
llm = OpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"], temperature=0.6)

# Setting up chat memory
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

#initialising the chat

# Setting up chat engine
chat_engine = index.as_chat_engine(
    chat_mode="context",
    llm=llm,
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about data related to Nigeria. If you are asked anything out of context, just say you don't know"
        "you were trained on different data from the NBS"
    ),
)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H-%M-%S')
        if isinstance(obj, HumanMessage):
            return obj.__dict__  # Serialize HumanMessage object as its dictionary representation
        return json.JSONEncoder.default(self, obj)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H-%M-%S')
        if isinstance(obj, AIMessage):
            return obj.__dict__  # Serialize HumanMessage object as its dictionary representation
        return json.JSONEncoder.default(self, obj)

app.json_encoder = CustomJSONEncoder


@app.route('/')
def home():
    return render_template('index.html'), 200

# Dictionary to store chat history for each user
user_chat_history = {}

@app.route('/chat', methods=['POST'])
def chat():
    input = None
    username = request.args.get('username', '') #query parameter to fetch username
    if request.headers['Content-Type'] == 'application/json':
        question = request.get_json()['input']
    else:
        question = request.form.get('input', '')
    
    #initializing the chat
    if 'flow' not in session:
        session['flow'] = [
            {'content':  """ 
             welcome the user with their {username}. You are a chatbot, able to have normal interactions, as well as talk 
             about data related to Nigeria. If you are asked anything out of context, just say you don't know
             you were trained on different data from the NBS  """},
             {'content':question}
             ]
    # Get or create chat history for the user
    if username not in user_chat_history:
        user_chat_history[username] = []

    timestamp=datetime.now()

    user_chat_history[username].append(HumanMessage(content=question, timestamp=timestamp))
    response = chat_engine.chat(question)

    

    # Convert the response to string if it's not already
    if not isinstance(response, str):
        response = str(response)

    user_chat_history[username].append(AIMessage(content=response, timestamp=timestamp))

    # Log the question and response
    logging.info(f"User question: {question}")
    logging.info(f"AI response: {response}")

    messages = []
    for msg in user_chat_history[username]:
        if isinstance(msg, HumanMessage):
            messages.append({'role': username, 'content': msg.content, 'timestamp': msg.timestamp.strftime('%Y-%m-%d %H-%M-%S')})
        elif isinstance(msg, AIMessage):
            messages.append({'role': 'ai', 'content': msg.content, 'timestamp': msg.timestamp.strftime('%Y-%m-%d %H-%M-%S')})

    return jsonify({'response': response, 'messages': messages})


# Store chat history per session ID
def store_chat_history(username, messages):
    user_chat_history[username] = messages
    
# Retrieve chat history based on session ID
def retrieve_chat_history(username):
    return user_chat_history.get(username, [])

@app.route('/history', methods=['GET'])
def history():
    username = request.args.get('username', '')  # Extract username from query parameter
    # Retrieve chat history for this session ID
    chat_history = retrieve_chat_history(username)
    
    # Convert HumanMessage objects to dictionaries
    formatted_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_message = {
                'role': username,
                'content': message.content,
                'timestamp': message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            formatted_history.append(formatted_message)
        elif isinstance(message, AIMessage):
            formatted_message = {
                'role': 'ai',
                'content': message.content,
                'timestamp': message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            formatted_history.append(formatted_message)

    return jsonify({'chat_history': formatted_history})


if __name__ == '__main__':
    app.run(debug=debug_mode, host=host_no, port=port_no)
             