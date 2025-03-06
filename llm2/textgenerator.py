from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

oai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

#from openai import OpenAI

#client = OpenAI(api_key=api_key)
store = {}  # memory is maintained outside the chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_chat_chain(session_id, model_name = "openai" , streaming=False):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
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
                "You are a helpful travel assistant.",
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

@app.route('/generate_text', methods=['POST'])
def predict():
    logging.info("Request received")
    data = request.get_json()
    input_message = data['in_message']
    session_id = data.get('session_id', 'default')
    chain = get_chat_chain(session_id)
    output = chain.run(input_message)
    logging.info(output)
    logging.info(output.content)
    return jsonify({'Generated Response': output.content})


@app.route('/chatresponse', methods=['POST'])
def chatresponse():
    logging.info("Request received")
    input_message = request.form['in_message']
    session_id = request.form['session_id']
    model_name = request.form['model_name']
    config = {"configurable": {"session_id": session_id}}
    chain = get_chat_chain(session_id, model_name)
    output = chain.invoke(
                {"messages": [HumanMessage(content=input_message)]}
                , config= config
            )
    logging.info(output)
    history = chain.get_session_history(session_id)
    messages_history = []
    for message in history:
        logging.info(message)
        detlist = message[1]
        for detail in detlist:
            logging.info(detail)
            if(type(detail) == AIMessage):
                messages_history.append(("Chatbot", detail.content))
            elif(type(detail) == HumanMessage):
                messages_history.append(("Member", detail.content))
            else:
                messages_history.append((type(detail), detail.content))
            
    #messages_history.reverse()
    logging.info(messages_history)
    #logging.info(output.content)
    response_text = output
    return render_template("inputtest.html", **locals())



@app.route('/', methods=['GET'])
def main():

    return render_template("inputtest.html")

if __name__ == '__main__':
    app.run(debug=True)