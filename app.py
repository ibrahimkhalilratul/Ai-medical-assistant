import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import re

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def extract_topic(text):
    """Extracts the main topic from the user input."""
    keywords = re.findall(r'\b(?:hormones|lungs|heart|brain|digestion|immune system)\b', text, re.IGNORECASE)
    return keywords[0].capitalize() if keywords else text[:50] + "..."

def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide")
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'saved_responses' not in st.session_state:
        st.session_state.saved_responses = {}
    if 'archived_responses' not in st.session_state:
        st.session_state.archived_responses = {}
    if 'selected_response' not in st.session_state:
        st.session_state.selected_response = None
    
    st.sidebar.title("Chat History")
    to_delete = None
    to_archive = None
    selected_topic = None
    
    for entry in reversed(st.session_state.history):
        with st.sidebar.expander(entry):
            if st.button("Open", key=f"open_{entry}"):
                st.session_state.selected_response = st.session_state.saved_responses.get(entry, "No saved response.")
            if st.button("Delete", key=f"delete_{entry}"):
                to_delete = entry
            if st.button("Archive", key=f"archive_{entry}"):
                to_archive = entry
    
    if to_delete:
        st.session_state.history.remove(to_delete)
        st.session_state.saved_responses.pop(to_delete, None)
        st.experimental_rerun()
    
    if to_archive:
        st.session_state.archived_responses[to_archive] = st.session_state.saved_responses.pop(to_archive, None)
        st.session_state.history.remove(to_archive)
        st.experimental_rerun()
    
    st.title(" Your Medical Assistant")
    st.write("Ask any medical-related question.")
    
    chat_container = st.container()
    prompt = st.chat_input("Type your message here and press Enter:")
    
    if prompt:
        topic = extract_topic(prompt)
        if topic not in st.session_state.history:
            st.session_state.history.append(topic)  # Store topic in history
        
        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question. 
            I gave you the huge range of medical data so that you can act like a medical teacher or as a doctor.
            If you don't know the answer, just say that you don't know, try to make up an answer realterd to the context and the semilar questions.
            Try to understand the query and try to provide the answer in the context of the query.
            Try to give an ans with breakdown and whenever it comes to the medical terms, try to explain them in layman's terms.
            Prescribe the medicine or the treatment from the given context. And list out the symptoms and the causes.
            Also list out the side effects of the medicine and the treatment. And suggested medicines and the treatment.
            And don't say that from the given context,or from the given information, or from the given data.
            Don't provide anything out of the given context.
        
            Context: {context}
            Question: {question}
        
            Start the answer. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "No result found.")
            st.session_state.saved_responses[topic] = result  # Store response linked to topic

            with chat_container:
                st.chat_message("user").markdown(prompt)
                st.chat_message("assistant").markdown(result)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.session_state.selected_response:
        st.chat_message("assistant").markdown(st.session_state.selected_response)

if __name__ == "__main__":
    main()
