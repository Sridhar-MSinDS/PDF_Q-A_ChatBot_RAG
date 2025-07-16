import warnings
warnings.filterwarnings('ignore')

# !pip install streamlit PyPDF2 langchain langchain_openai langchain_community

# --------------------------------------
# ✅ STEP 1: Import Required Libraries
# --------------------------------------
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------
# ✅ STEP 2: Streamlit App Layout
# --------------------------------------
st.header("📄 PDF Chatbot - Ask Questions from uploaded PDF")

with st.sidebar:
    st.title("📚 Q&A from PDF")
    file = st.file_uploader("Upload a PDF", type="pdf")

# --------------------------------------
# ✅ STEP 3: Load and Extract Text from PDF
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
    # ✅ STEP 5: Create Embeddings using OpenAI
    # --------------------------------------
    OpenAI_API_KEY = "Your OpenAI API key"
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    # --------------------------------------
    # ✅ STEP 6: Store Embeddings in FAISS Vector Store
    # --------------------------------------
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --------------------------------------
    # ✅ STEP 7: Take User Query and Perform Similarity Search
    # --------------------------------------
    user_query = st.text_input("💬 Ask a question based on the PDF")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # --------------------------------------
        # ✅ STEP 8: Define LLM and Create Q&A Chain
        # --------------------------------------
        llm = ChatOpenAI(api_key=OpenAI_API_KEY, model="gpt-3.5-turbo", temperature=0, max_tokens=300)

        prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent and helpful assistant specialized in summarizing and answering questions from study notes.

    Use the provided context extracted from the user's PDF notes to answer the question accurately and concisely.

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
        # ✅ STEP 9: Run Chain and Display Output
        # --------------------------------------
        output = chain.invoke({
            "input": user_query,
            "input_documents": matching_chunks
        })

        st.subheader("🧠 Answer")
        st.write(output)