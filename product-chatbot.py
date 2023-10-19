import json
import os
import streamlit as st
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

    os.environ['OPENAI_API_KEY'] = st.secrets["openai_key"]
    llm = OpenAI(temperature=0)
    openai_embeddings = OpenAIEmbeddings()
    table_name = 'pdf_q_n_a_table_3'
    keyspace = "vector_preview"

    pdf_index = Cassandra(
            embedding = openai_embeddings,
            session = session,
            keyspace = keyspace,
            table_name = table_name,
        )
    
    if "pdf_index" not in st.session_state:
        st.session_state.pdf_index = pdf_index
    st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            index_placeholder = st.session_state.pdf_index
            pdf_response = index_placeholder.query_with_sources(prompt, llm = llm)
            cleaned_response = pdf_response["answer"]
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": cleaned_response})


if __name__ == '__main__':
    main()
