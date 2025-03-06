from flask import Flask, request, Response, jsonify, render_template
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

def get_chat_chain(session_id, model_name = "openai", streaming=False):
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




@app.route('/chatresponsestream')
def chatresponsestream():
    logging.info("Request received")
    input_message = request.args.get('in_message', 'default')
    session_id = request.args.get('session_id', 'default')
    model_name = request.args.get('model_name', 'openai')
    config = {"configurable": {"session_id": session_id}}
    return Response(generate_response(session_id, model_name, input_message), content_type='text/event-stream')

def generate_response(session_id, model_name, input_message):
    chain = get_chat_chain(session_id, model_name, streaming=True)

    for chunk in chain.stream({"messages": [HumanMessage(content=input_message)]}, config={"configurable": {"session_id": session_id}}):
        #chunk_text = chunk.get('response', '').replace("\n", "<br>")
        if(chunk.strip()):
            yield f"data: {chunk}\n\n"


    yield f"data: [DONE] \n\n"


@app.route('/', methods=['GET'])
def main():

    return render_template("streaming_output.html")

if __name__ == '__main__':
    app.run(debug=True)