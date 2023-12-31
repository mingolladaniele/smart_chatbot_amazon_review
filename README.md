# Amazon Fashion Review Chatbot
<p align="center">
  <img src="src/gui.png" alt="Chatbot's GUI" style="width: 70%; margin: 0 auto;">
</p>

# Description
This project involves building an intelligent chatbot web app to analyze Amazon Fashion reviews. The chatbot provides a natural and intuitive conversational experience, allowing users to explore customer reviews, identify trends, and understand what makes a great customer experience.

# Data
The data used for this project can be found at this [link](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews). The interface allow you to upload multiple `json` files up to a maximum of 200 mb.

# Architecture
<p align="center">
  <img src="src/architecture.png" alt="Chatbot's Architecture" style="width: 70%; margin: 0 auto;">
</p>

1. **Text Splitting:** This initial step involves segmenting the input text into smaller, more manageable portions. Employing a character-based text splitter, the text is divided into coherent units, facilitating better processing.

2. **Embedding with OpenAI:** Once split, the text chunks undergo embedding using OpenAI's embedding techniques. These embeddings encapsulate the underlying semantic essence, converting the textual content into numerical representations that encapsulate both context and meaning.

3. **Vector Store Creation:** The embedded text chunks are aggregated and housed within a vector store. This data structure is meticulously designed for proficient similarity searches. Making use of the FAISS library, which specializes in similarity search and clustering, the vector store ensures optimized search operations.

4. **User Input and OpenAI Embedding:** When a user poses a question, the input is seamlessly embedded using OpenAI's embedding approach. This transformation prepares the user's query for further processing and analysis.

5. **Semantic Search and Matching:** The newly embedded user query is evaluated against the embedded text chunks previously stored in the vector store. A semantic search and matching process ensues, identifying the text chunks that most closely correspond to the user's query based on their embedded representations.



# Setup
1. Install Docker on your system if not already installed.
2. Open a terminal and navigate to the root folder of the project.
3. Create a `.env` file within the `chatbot` directory and insert this line in it: `OPENAI_API_KEY = <your_api>`.
4. Execute the command `docker-compose up`.
5. Access the chatbot in your web browser at `http://localhost:8501`.