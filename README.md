# 📄 ChatBot – PDF Q&A Chatbot using LangChain, OpenAI, and Streamlit

This ChatBot is an AI-powered chatbot that allows users to upload any PDF (like study notes, documents, or reports) and ask questions about its content. 
It uses LangChain, OpenAI's GPT-3.5, and FAISS for intelligent retrieval-based question answering, and is built using Streamlit for an interactive web interface.

# 🚀 Features

🔍 Extracts and understands content from any PDF.

🤖 Answers questions using GPT-3.5-turbo based on PDF content.

🧠 Uses OpenAI embeddings + FAISS vector store for semantic search.

📚 Ideal for summarizing, studying, or querying long documents.

💻 Simple UI using Streamlit.

# 🛠️ Tech Stack

Tool	Purpose
Streamlit	UI for uploading PDFs and entering queries
PyPDF2	PDF text extraction
LangChain	Q&A chain and prompt templates
OpenAI API	GPT-3.5-turbo LLM & Embeddings
FAISS	Fast vector similarity search

# 📂 Project Structure

├── app.py                # Streamlit application

├── requirements.txt      # Dependencies

├── README.md             # This file

🧠 How It Works (Architecture)
User uploads a PDF via Streamlit sidebar.

Text is extracted using PyPDF2.

Text is chunked into small overlapping parts for better context handling.

Embeddings are created for each chunk using OpenAI's Embedding API.

FAISS vector store is used to store and retrieve semantically relevant chunks.

User enters a query.

The relevant chunks are retrieved and passed to GPT-3.5 using LangChain's Q&A chain.

The answer is displayed in the Streamlit app.

# 🧾 Setup Instructions
✅ Prerequisites:
Python 3.8+

OpenAI API key (get from https://platform.openai.com)

## 🔧 1. Clone the repository:
bash

git clone https://github.com/yourusername/pdf-qa-chatbot.git
cd pdf-qa-chatbot

## 📦 2. Install dependencies:
bash

pip install -r requirements.txt
If requirements.txt is missing, install manually:

bash

pip install streamlit langchain openai PyPDF2 faiss-cpu

## 🔑 3. Add your OpenAI API key:

In app.py, replace:

python

OpenAI_API_KEY = "Your OpenAI API key"
with your real API key.

Alternatively, use environment variables for security:

python

import os
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

## ▶️ 4. Run the Streamlit app:
bash

streamlit run app.py
✨ Example Usage
Click "Browse files" in the sidebar to upload a PDF.

Ask a question like:

"What are the key points in chapter 2?"

"Explain the difference between X and Y?"

The bot will fetch context and generate a smart answer!

# 🧪 Example Prompt Used

text

You are an intelligent and helpful assistant specialized in summarizing and answering questions from study notes.
Use the provided context extracted from the user's PDF notes to answer the question accurately and concisely.
If the answer is not available in the context, say: "I'm sorry, I couldn't find that information in the uploaded PDF."

# ✅ To-Do / Improvements

 Add support for multi-page PDF display

 Cache embeddings to reduce re-computation

 Add UI theme customization

 Save chat history per session

# 📄 License

This project is licensed under the MIT License.

# 👋 Acknowledgments

LangChain

OpenAI

Streamlit

FAISS
