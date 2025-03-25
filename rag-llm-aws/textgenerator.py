from flask import Flask, request, Response, jsonify, render_template, session
from flask_login import login_user, current_user, logout_user, login_required, LoginManager
import joblib
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
import pymongo
from pymongo.mongo_client import MongoClient

from pymongo.server_api import ServerApi
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
import pandas as pd
import asyncio
from sagemaker_llm.process_request import process_async_completion
from sagemaker_llm.LineIterator import LineIterator
# Import necessary libraries
import json

from models.user import User
from util.chats import Chats

from util.movies import vector_search

# Initialize Flask app
app = Flask(__name__)
app.secret_key = b'__5#y2L"F4Q8z\n\xec]/'
# Configure Flask-Login
app.config['SECRET_KEY'] = b'__5#y2L"F4Q8z\n\xec]/'
login_manager = LoginManager()
login_manager.init_app(app)
# Load environment variables from .env file
load_dotenv()

oai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Load environment variables from .env file
load_dotenv()
uri = os.getenv("MONGO_DB_CONNECTION")
client = MongoClient(uri, server_api=ServerApi('1'))
#from openai import OpenAI

#client = OpenAI(api_key=api_key)
store = {}  # memory is maintained outside the chain
mongo_uri = os.getenv('MONGO_DB_CONNECTION')

movie_db = client['movies']
movie_collection = movie_db['movie_collection'] 

@app.route('/', methods=['GET'])
def main():
    logout_user()
    session.clear()
    return render_template("index.html")


@app.route('/login', methods=['POST'])
def userlogin():
    username= request.form["username"]
    logging.info("Request received:{username}")
    user = User(username)
    Chats.retreive_user_details(client, username)
    chats_df = Chats.retreive_user_chats(client, username)
    if(chats_df.empty):
        logging.info("No chats found")
        chats = []
        chat_details = []
    else:
        chats = chats_df.values.tolist()

        first_chat = chats_df.iloc[0] #to access first row
        logging.info(f"getting details for first chat: {first_chat}")
        chat_id = first_chat.chat_id
        chat = Chats()
        chat_details_df = chat.retreive_user_chat_details(client, username, chat_id)
        chat_details = populate_chat_details_to_memory(username, chat_id)


    logging.info(f"User chats: {chats}")
    login_user(user, remember=True)
    return render_template("streaming_output.html", **locals())

@app.route('/refresh')
def refresh():
    username = current_user.username
    logging.info("Request received:{username}")
    user = User(username)
    Chats.retreive_user_details(client, username)
    chats_df = Chats.retreive_user_chats(client, username)
    if(chats_df.empty):
        logging.info("No chats found")
        chats = []
        chat_details = []
    else:
        chats = chats_df.values.tolist()

        first_chat = chats_df.iloc[0] #to access first row
        logging.info(f"getting details for first chat: {first_chat}")
        chat_id = first_chat.chat_id
        chat = Chats()
        chat_details_df = chat.retreive_user_chat_details(client, username, chat_id)
        chat_details = populate_chat_details_to_memory(username, chat_id)    

    return render_template("streaming_output.html", **locals())


@app.route('/logout')
def userlogout():
    logout_user()
    session.clear()
    return render_template("index.html")


@app.route('/newchat')
def newchat():
    username = current_user.username
    logging.info("Request received:{username}")
    user = User(username)
    Chats.retreive_user_details(client, username)
    chats_df = Chats.retreive_user_chats(client, username)
    chat_details = []
    if(chats_df.empty):
        logging.info("No chats found")
        chats = []
        chat_details = []
    else:
        new_row = pd.DataFrame([{"_id": "new_chat_id","chat_id": "new_chat_id", "chat_title":"New chat", "username": "testlogin"
                                 ,"created_at":pd.Timestamp.now(),"updated_at":pd.Timestamp.now()}], index=[0])
        chats_df = pd.concat([new_row, chats_df], ignore_index=True)
        chats = chats_df.values.tolist()
        logging.info(f"User chats: {chats}")

    return render_template("streaming_output.html", **locals())



@app.route('/chatdetails')
def chatdetails():
    username = current_user.username
    chat_id = request.args.get('chatid', 'default')
    logging.info("Request received:{username}")
    user = User(username)
    Chats.retreive_user_details(client, username)
    chats_df = Chats.retreive_user_chats(client, username)
    if(chats_df is not None):
        logging.info(f"getting details for chat: {chat_id}")
    chats = chats_df.values.tolist()
    chat_details = populate_chat_details_to_memory(username, chat_id)
    logging.info(f"User chats: {chats}")
    login_user(user, remember=True)
    return render_template("streaming_output.html", **locals())


def populate_chat_details_to_memory(username, chat_id):
    chat = Chats()
    chat_details_df = chat.retreive_user_chat_details(client, username, chat_id)
    if(chat_details_df.empty):
        logging.info("No chat details found")
        chat_details = []        
    else:
        logging.info(f"Chat details: {chat_details_df}")
        chat_details_df = chat_details_df.drop(columns=['_id', 'chat_id'])
        chat_details_df = chat_details_df.sort_values(by='created_at')
        logging.info(f"Chat details: {chat_details_df}")
        if(chat_details_df.empty):
            logging.info("No chat details found")
            chat_details = []
        else:                
            chat_details = chat_details_df.values.tolist()
            chat_history = get_session_history(chat_id)
            for chat_del in chat_details:
                inm = chat_del[2]
                role = chat_del[1]
                logging.info(f"Chat row details: {chat_del}")
                
                if(role == "user"):
                    message = HumanMessage(content=inm)
                else:
                    message = AIMessage(content=inm)
                chat_history.add_message(message)
            logging.info(f"Chat details: {chat_history}")
                
        #logging.info(f"Chat details: {chat_details}") 
    return chat_details




@app.route('/chatresponsestream')
def chatresponsestream():
    logging.info("Request received")
    input_message = request.args.get('in_message', 'default')
    chat_id = request.args.get('chat_id', 'default')
    if(chat_id == 'new_chat_id' or chat_id == 'default' or chat_id == ''):
        chat_id = None
    model_name = request.args.get('model_name', 'openai')
    rag =  request.args.get('rag', 'no')
    config = {"configurable": {"chat_id": chat_id}}
    chat = Chats()
    username=current_user.username
    logging.info(f"Username: {username}, chat_id: {chat_id}")
    chat_id = chat.create_user_chat(client, username=username, chat_id = chat_id, chat_content=input_message, chat_title=input_message, role="user")
    if(model_name == "fb_llama"):
        return Response(generate_response_llama(input_message, username, chat_id, rag), content_type='text/event-stream')
    else:
        return Response(generate_response(model_name, input_message, username, chat_id, rag), content_type='text/event-stream')

def generate_response(model_name, input_message, username, chat_id, rag):
    chain = get_chat_chain(chat_id, username, model_name, streaming=True)
    if(rag == 'yes'):
        get_knowledge = vector_search(input_message, movie_collection)

        search_result = ''
        logging.info(f"Recommendations begin\n\n")
        for result in get_knowledge:
            search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('plot', 'N/A')}\\n"
            logging.info(f"movies:Title: {result.get('title', 'N/A')}, Plot: {result.get('plot', 'N/A')}\\n")
        
        logging.info(f"Recommendations done\n\n")
        modified_message = "Answer this user query: " + input_message + " with the following context: " + search_result
    else:
        modified_message = "Answer this user query: " + input_message
    data = ""
    for chunk in chain.stream({"messages": [HumanMessage(content=modified_message)]}, config={"configurable": {"session_id": chat_id}}):
        #chunk_text = chunk.get('response', '').replace("\n", "<br>")
        if(chunk.strip()):
            data += chunk
            yield f"data: {chunk}\n\n"
    chat = Chats()
    chat.create_user_chat(client, username=username, chat_content=data, chat_id=chat_id, role="AI Chat")
    logging.info(f"Final data: {data}")
    yield f"data: [DONE] \n\n"


def populate_message_history_llama(username, chat_id):
    chat = Chats()
    chat_details_df = chat.retreive_user_chat_details(client, username, chat_id)
    if(chat_details_df.empty):
        logging.info("No chat details found")
        chat_details = []        
    else:
        logging.info(f"Chat details: {chat_details_df}")
        chat_details_df = chat_details_df.drop(columns=['_id', 'chat_id'])
        chat_details_df = chat_details_df.sort_values(by='created_at')
        logging.info(f"Chat details: {chat_details_df}")
        if(chat_details_df.empty):
            logging.info("No chat details found")
            chat_details = []
        else:                
            chat_details = chat_details_df.values.tolist()
            chat_history = get_session_history(chat_id)
            history = ""
            for chat_del in chat_details:
                inm = chat_del[2]
                role = chat_del[1]
                logging.info(f"Chat row details: {chat_del}")
                
                if(role == "user"):
                    hismessage = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inm}<|eot_id|>"
                else:
                    hismessage = f"<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n{inm}<|eot_id|>"
                history += hismessage
            return history
    return ""


def generate_response_llama(input_message, username, chat_id, rag):
    #chain = get_chat_chain(chat_id, username, , streaming=True)
    prompt_message = ""
    if(rag == 'yes'):
        get_knowledge = vector_search(input_message, movie_collection)

        search_result = ''
        logging.info(f"Recommendations begin\n\n")
        for result in get_knowledge:
            search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('plot', 'N/A')}\\n"
            logging.info(f"movies:Title: {result.get('title', 'N/A')}, Plot: {result.get('plot', 'N/A')}\\n")
        
        logging.info(f"Recommendations done\n\n")
        prompt_message = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nyou are a helpful assistant. Answer user query with the following context: " + search_result + "<|eot_id|>"
    else:
        prompt_message = "" #"<|start_header_id|>system<|end_header_id|>\n\nAlways answer with emojis<|eot_id|>"
    history = populate_message_history_llama(username, chat_id)
    logging.info(f"History: {history}")
    logging.info(f"Recommendations done\n\n")
    modified_message = "Answer this user query: " + input_message + " with the following context: "
    mdata = ""

    sample_prompt = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    #asyncio.run((process_sync_completion(sample_prompt)))

    response = asyncio.run(process_async_completion(history + prompt_message + sample_prompt))

    response_parsed = True       
    start_json = b'{'
    stop_token = '<|eot_id|>'
    stop_token_found = False
    number_of_words = 0
    start_header_pos = -1
    assistant_pos = -1
    end_header_pos = -1
    ignore_response_list = ['<|start_header_id|>', 'assistant', '<|end_header_id|>']
    full_resp = ''
    number_of_message = 0
    response_parsed = True
    complete_response = ""
    for line in LineIterator(response):
        #print(type(line))
        #print(line)
        if line != b'': # and "data" in line: #start_json in line:
            try:
                if("data" in line):
                    data = json.loads(line).decode('utf-8')
                else:
                    data = json.loads(line.replace("data:", "")) #json.loads(line[line.find(start_json):].decode('utf-8'))
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                continue
            # print(data)
            if data['token']['text'] != stop_token and stop_token_found == False:
                #print(data['token']['text'],end='')
                part = data['token']['text']
                number_of_words = number_of_words + 1
                full_resp += part
                if(part.find("start_header_id") > 0):
                    number_of_message+= 1
                    if(number_of_message>1):
                        stop_token_found = True
                if(part in ignore_response_list):
                    continue
                if(part.find("<|eot_id|>") > 0):
                    stop_token_found = True
                if(number_of_words > 3 and part.find("start_header_id") >-1):
                    start_header_pos = number_of_words
                if(part.find("assistant") > -1):
                    assistant_pos = number_of_words
                if(part.find("end_header_id") > -1):
                    end_header_pos = number_of_words
                #print(f'{part.find("start_header_id")}, {part.find("assistant")}, {part.find("end_header_id")}')
                if(number_of_words > 3 and (assistant_pos == start_header_pos+1) and (end_header_pos == start_header_pos+2)):
                    stop_token_found = True
                #print(number_of_words)
                #print(part)
                #logging.info(f"print response--> {part}")
                mdata += part
                yield f"data: {part}\n\n"

                complete_response+=part
            if(stop_token_found == True):
                break
        if(stop_token_found == True):
            break
    chat = Chats()
    chat.create_user_chat(client, username=username, chat_content=mdata, chat_id=chat_id, role="AI Chat")
    logging.info(f"Final data: {mdata}")
    yield f"data: [DONE] \n\n"


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)



class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


def get_session_history(conversation_id: str) -> BaseChatMessageHistory:
    if (conversation_id) not in store:
        store[conversation_id] = InMemoryHistory()
    return store[conversation_id]


def get_chat_chain(chat_id, username, model_name = "openai", streaming=False):
    if (username, chat_id) not in store:
        store[(username, chat_id)] = InMemoryHistory()
    
    if model_name == "openai":
        llm = ChatOpenAI(api_key=oai_api_key, streaming=streaming, model_name = "gpt-4o-mini" )
    elif model_name == "groq":
        llm = ChatGroq(api_key=groq_api_key, streaming=streaming, model_name = "llama-3.3-70b-versatile" )
    elif model_name == "gemini":
        llm = ChatGoogleGenerativeAI(google_api_key=gemini_api_key, streaming=streaming, model = "gemini-2.0-flash" )        
    logging.info(f"Using model: {model_name}")

    # Initialize the parser once
    parser = StrOutputParser()
    # System and user message templates
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie recommendation system.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create a chain that includes the prompt and the LLM
    chain = prompt | llm | parser

    # Create a runnable (executable) with message history
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )

    
    return with_message_history





if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5001", debug=True)