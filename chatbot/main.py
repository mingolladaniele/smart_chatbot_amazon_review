import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# TODO: this db for text embeddings is not permanent
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

def get_json_text(json_docs):
    df_list = []
    for j_file in json_docs:
        df = pd.read_json(j_file, lines=True)
        st.write(df[0:1000])
        # TODO: could be usefull to add as col the value in the col 'style'
        df = df[['overall', 'reviewTime', 'reviewText', 'summary']]
        df_list.append(df)
    final_df = pd.concat(df_list)
    return final_df.to_string(header=False)
        
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= '\n'
        # TODO: to change, check length each line
        # n chars in a chunk
        # how many chars to consider since the end of the previous chunk -> avoiding losing context
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts= text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # TODO: understand it better
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    for i, msg in enumerate(response['chat_history']):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot for Amazon Reviews", page_icon='🤖')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChatBot for Amazon Reviews 🤖")
    user_question = st.chat_input("Say something")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        json_docs = st.file_uploader("Upload your JSON here and click on 'Process'", accept_multiple_files=True, type=['json'])
        # TODO: add error msg if usr doesnt upload a file
        if json_docs:
            # button shows only after file uploded
            if st.button("Process"):
                with st.spinner("Processing"):      
                    # get the json text
                    raw_text = get_json_text(json_docs)
                    text_chunks = get_text_chunks(raw_text)
                    # create vector store
                    vector_store = get_vector_store(text_chunks)
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()