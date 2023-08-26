import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def load_json_docs(json_docs):
    """
    Load JSON documents and concatenate relevant columns into a DataFrame.

    Args:
        json_docs (list): List of JSON docs.

    Returns:
        pandas.DataFrame: Concatenated DataFrame with relevant columns.
    """
    df_list = []
    for j_file in json_docs:
        df = pd.read_json(j_file, lines=True)
        df = df[["overall", "reviewText"]]
        df_list.append(df)
    final_df = pd.concat(df_list)
    return final_df


def split_text_into_chunks(text):
    """
    Split input text into smaller chunks using a character-based splitter.

    Args:
        text (str): Input text.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(separator="\n")
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    """
    Create a vector store from input text chunks using OpenAI embeddings.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        langchain.vectorstores.FAISS: Vector store for text chunks.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
