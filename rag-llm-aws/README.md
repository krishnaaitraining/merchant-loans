
Once you checkout the code, Please add ".env file with the following properties:
<br>
OPENAI_API_KEY
<br>
GROQ_API_KEY
<br>
GEMINI_API_KEY
<br>
MONGO_DB_CONNECTION
<br>
PINECONE_KEY
<br>
AWS_ACCESS_KEY_ID<br>
AWS_SECRET_ACCESS_KEY<br>
AWS_REGION<br>
AWS_SM_ENDPOINT_NAME<br>
HUGGING_FACE_TOKEN<br>

<br>
After that run the code using: python textgenerator.py
<br>
This will start the application. If it runs successfully, open the URL http://127.0.0.1:5001/ in the browser.


sudo yum update -y
sudo yum install python3 git -y
python -m ensurepip --upgrade
pip3 install -r requirments.txt
