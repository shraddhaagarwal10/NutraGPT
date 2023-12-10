from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
import os

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
HF_API_KEY = os.environ["HF_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')
index = pinecone.Index('nutraceuticlas-chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=10, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="gpt-3.5-turbo",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']


def query_refiner(conversation, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or the specific chat model you are using
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"CONVERSATION LOG:\n{conversation}\n\nQuery: {query}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in query_refiner: {e}")
        return query






def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string