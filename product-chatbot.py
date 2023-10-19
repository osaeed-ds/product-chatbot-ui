import json
import os
import streamlit as st
import openai
from cassandra.query import SimpleStatement
import openai
import numpy
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, PyPDFLoader
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from copy import deepcopy
from tempfile import NamedTemporaryFile

@st.cache_resource
def create_datastax_connection():

    cloud_config= {'secure_connect_bundle': 'secure-connect-osaeed-vector.zip'}

    CLIENT_ID = "token"
    CLIENT_SECRET = st.secrets["astra_token"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

def main():

    index_placeholder = None
    !curl -o secure-connect-osaeed-vector.zip 'https://datastax-cluster-config-prod.s3.us-east-2.amazonaws.com/8dce574d-98e5-4b4d-bdfe-2477340a5d09-1/secure-connect-osaeed-vector.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2AIQRQ76S2JCB77W%2F20231019%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20231019T082945Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=291297335ac710bbea58ceb6fd3e2613470d3bed799412848d3977501d810787'

    st.set_page_config(page_title = "Product Catalog Search", page_icon="📔")
    st.header('📔 Product Catalog Search')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    session = create_datastax_connection()
    model_id = "text-embedding-ada-002"

    os.environ['OPENAI_API_KEY'] = st.secrets["openai_key"]
    openai.api_key = st.secrets["openai_key"]
    llm = OpenAI(temperature=0)
    openai_embeddings = OpenAIEmbeddings()
    table_name = 'products_table'
    keyspace = "vector_preview"

    st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("What kind of product are you looking for?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            embedding = openai.Embedding.create(input=prompt, model=model_id)['data'][0]['embedding']
            query = SimpleStatement(
                """
                SELECT *
                FROM {}.products_table
                ORDER BY openai_description_embedding ANN OF {} LIMIT 5;
                """.format(keyspace,embedding)
            )
            results = session.execute(query)
            top_5_products = results._current_rows
            message_objects = []
            message_objects.append({"role":"system",
                        "content":"You're a chatbot helping customers with questions and helping them with product recommendations"})

            message_objects.append({"role":"user",
                        "content": prompt})

            message_objects.append({"role":"user",
                        "content": "Please give me a detailed explanation of your recommendations"})

            message_objects.append({"role":"user",
                        "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})

            message_objects.append({"role": "assistant",
                        "content": "I found these 3 products I would recommend"})

            products_list = []

            for row in top_5_products:
                brand_dict = {'role': "assistant", "content": f"{row.description}"}
                products_list.append(brand_dict)

            message_objects.extend(products_list)
            message_objects.append({"role": "assistant", "content":"Here's my summarized recommendation of products, and why it would suit you:"})

            completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=message_objects
            )            
            cleaned_response = completion.choices[0].message['content']
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": cleaned_response})


if __name__ == '__main__':
    main()
