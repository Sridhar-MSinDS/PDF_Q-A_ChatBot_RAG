# --------------------------------------
# ✅ STEP 1: Import Required Libraries
# --------------------------------------
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq  # 👈 For Groq LLM

# --------------------------------------
# ✅ STEP 2: Streamlit App Layout
# --------------------------------------
st.header("📄 PDF Chatbot - Ask Questions from Uploaded PDF")

with st.sidebar:
    st.title("📚 Upload Your PDF")
    file = st.file_uploader("Upload a PDF file", type="pdf")

# --------------------------------------
# ✅ STEP 3: Extract Text from PDF
# --------------------------------------
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # --------------------------------------
    # ✅ STEP 4: Split Text into Chunks
    # --------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # --------------------------------------
    # ✅ STEP 5: Generate Embeddings using OpenAI
    # --------------------------------------
    OPENAI_API_KEY = "your-openai-api-key"  # 🔐 Replace with your OpenAI key
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # --------------------------------------
    # ✅ STEP 6: Store Embeddings in FAISS
    # --------------------------------------
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --------------------------------------
    # ✅ STEP 7: Get User Query & Retrieve Chunks
    # --------------------------------------
    user_query = st.text_input("💬 Ask a question from your PDF")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # --------------------------------------
        # ✅ STEP 8: Define Groq LLM and Prompt
        # --------------------------------------
        GROQ_API_KEY = "your-groq-api-key"  # 🔐 Replace with your Groq API key
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="mixtral-8x7b-32768",  # You can also use "llama3-70b-8192"
            temperature=0,
            max_tokens=300
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are an intelligent and helpful assistant specialized in summarizing and answering questions from study notes.

            Use the provided context extracted from the user's PDF to answer the question accurately and concisely.

            If the answer is not available in the context, say: "I'm sorry, I couldn't find that information in the uploaded PDF."

            ---------------------
            Context:
            {context}

            Question:
            {input}
            ---------------------
            Answer:
            """
        )

        chain = create_stuff_documents_chain(llm, prompt)

        # --------------------------------------
        # ✅ STEP 9: Generate and Display Answer
        # --------------------------------------
        output = chain.invoke({
            "input": user_query,
            "input_documents": matching_chunks
        })

        st.subheader("🧠 Answer")
        st.write(output)
