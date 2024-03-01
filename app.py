import os
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPEN_AI_KEY"))

from flask import Flask, render_template, request, jsonify
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

# Loading environment variables from .env file
load_dotenv()

# Accessing environment variables
api_key = os.getenv('OPEN_AI_KEY')
print("API Key:", api_key)

app = Flask(__name__)


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

if 'flow' not in app.config:
        app.config['flow'] = [
            SystemMessage(
                content="You are a chatbot, able to have normal interactions, as well as talk"
        " about data related to Nigeria. If you are asked anything out of context, just say you don't know"
        "you were trained on different data from the NBS"          
                          )
        ]

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input = None
    if request.headers['Content-Type'] == 'application/json':
        question = request.get_json()['input']
    else:
        question = request.form.get('input', '')
    

    app.config['flow'].append(HumanMessage(content=question))
    response = chat_engine.chat(question)


    # Convert the response to string if it's not already
    if not isinstance(response, str):
        response = str(response)

    app.config['flow'].append(AIMessage(content=response))

    # Log the question and response
    logging.info(f"User question: {question}")
    logging.info(f"AI response: {response}")

    messages = []
    for msg in app.config['flow']:
        if isinstance(msg, HumanMessage):
            messages.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({'role': 'ai', 'content': msg.content})

    return jsonify({'response': response, 'messages': messages})


