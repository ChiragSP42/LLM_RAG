# Financial AI Helpbot

This is a small prototype is a 'Helpbot' built using Apple's OpenElm-450M model in conjunction with a local vector database to provide RAG capabilities and hence 'direct' the response of the bot to factually correct advice.

For this prototype I used Apple's `apple/OpenELM-450M-Instruct` from Huggingface. The choice for the model came as a suggestion from my professor as he had written a paper on it recently. For optimal performance I used the tokenizer that original developer's had used when training the model which is Meta's `meta-llama/Llama-2-7b-chat-hf`. For the vector database to act as our RAG element I chose ChromaDB as it was the only one that I found with a big community and it's open-source.

### Prerequisites:
1. Make sure all dependicies from 'requirements.txt' are downloaded. You can download all dependencies using the ```pip install -r requirements.txt```
2. Request access from Meta to allow the use of their tokenizer. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. Once access has been granted, create an access token by clicking on your Huggingface profile>Access Tokens>Create a new token.
4. Store the value of the token in your local .env file and save it as ``` HF_TOKEN```

## RAG

To build our vector database, we first need to collect relevant information related to the financial stock market. Luckily I came across a python library for wikipedia. It is a wrapper for Wikipedia's API so I could concentrate on the project rather than diverting resources to building a information retrieval system using their API.

Before running the main script, first run the `vector_database_builder.py`