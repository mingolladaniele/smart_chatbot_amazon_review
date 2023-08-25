import streamlit as st
from dotenv import load_dotenv
from utils.data_preprocessing import load_json_docs, split_text_into_chunks, create_vector_store
from utils.chatbot_functions import init_conversation_chain, handle_user_input


def init_settings():
    # Load API credentials & set up session state for conversation
    load_dotenv()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Configure Streamlit UI settings
    st.set_page_config(
        page_title="ChatBot for Amazon Reviews",
        page_icon="🤖",
        initial_sidebar_state="expanded",
    )

    st.header("Smart Chatbot for Amazon Product Reviews")
    st.caption(
        "This chatbot offers a natural conversational experience, helping users explore reviewer insights and learn about good reviews.\n\
        **Remember to upload your JSON files from the sidebar before asking questions!**"
    )
    st.divider()


def main():
    init_settings()
    with st.sidebar:
        st.subheader("YOUR DOCUMENTS")
        json_docs = st.file_uploader(
            "Upload your JSON files.\n\n**They all must have the same structure!**",
            accept_multiple_files=True,
            type=["json"],
        )
        if json_docs:
            if st.button("Process"):
                with st.spinner("Processing"):
                    df = load_json_docs(json_docs)
                    text_chunks = split_text_into_chunks(df.to_string(header=False))
                    vector_store = create_vector_store(text_chunks)
                    st.session_state.conversation = init_conversation_chain(
                        vector_store
                    )

    # Display chat input only if conversation state is initialized
    if st.session_state.conversation:
        user_question = st.chat_input(placeholder="Enter your question...")
        if user_question:
            handle_user_input(user_question, st.session_state.conversation)

if __name__ == "__main__":
    main()