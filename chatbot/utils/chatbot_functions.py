import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def init_conversation_chain(vector_store):
    """
    Initialize a conversational retrieval chain.

    Args:
        vector_store (langchain.vectorstores.FAISS): Vector store for text chunks.

    Returns:
        langchain.chains.ConversationalRetrievalChain: Initialized conversational chain.
    """
    # Initialize a ChatOpenAI language model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # Create a memory buffer to track conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question, conversation_chain):
    """
    Handle user input and display chatbot responses.

    Args:
        user_question (str): User's input question.
        conversation_chain (langchain.chains.ConversationalRetrievalChain): Conversation chain.

    Returns:
        None
    """
    # Use the conversation chain to generate a response
    response = conversation_chain({"question": user_question})
    # Display the conversation history and responses
    for i, msg in enumerate(response["chat_history"]):
        container_type = "user" if i % 2 == 0 else "assistant"
        container = st.chat_message(container_type)
        container.write(msg.content)
