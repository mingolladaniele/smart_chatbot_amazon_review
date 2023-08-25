import streamlit as st
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot for Amazon Reviews", page_icon='🤖')
    st.header("ChatBot for Amazon Reviews 🤖")
    st.text_input("Ask a question about your documents:")
    with st.sidebar:
        st.subheader("Your documents")
        json_docs = st.file_uploader("Upload your JSON here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            # get the json text
            # get the text
            # create vector store

if __name__ == '__main__':
    main()