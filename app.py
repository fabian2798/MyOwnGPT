from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from htmlTemplate import css, user_template, bot_template
from langchain.callbacks import get_openai_callback

def get_pdf_text(pdf_docs):
    # Extract the Text
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    # Split Text into Chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # 1000 characters per chunk
        chunk_overlap=200,  # each following chunks can contain 200 characters from previous chunk
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vector_store(chunks):
    # create Embeddings

    # using OpenAI Adav2
    # embeddings = OpenAIEmbeddings()

    # using own hardware to process data into chunks
    # running on gpu (default cpu)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={'device': 'gpu'})

    # create Vectorstore(Knowledgebase)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    #define laerge language model

    #track cost of using adav2 as embedding-model
    #with get_openai_callback() as cb:
        #llm = ChatOpenAI()
        #print(cb)

    llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-70b-chat-hf", model_kwargs={"temperature":0.5, "max_length":512})
    # occupy memory for conversations
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # create conversation_chain for follow-up questions
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    # create response

    # track cost of generated response when using openai
    #with get_openai_callback() as cb:
    #    response = st.session_state.conversation({'question': user_question})
    #    print(cb)

    response = st.session_state.conversation({'question': user_question})
    # complete chat history
    st.session_state.chat_history = response['chat_history']

    # handling output with template
    for i, message in enumerate(st.session_state.chat_history):
        # User Message
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        # AI Message
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # LOAD ENVIRONMENT VARIABLES
    load_dotenv()

    # Initialise session_state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Streamlit GUI Configuration
    st.set_page_config(page_title="Chat mit deinen PDF")
    st.header("Chat mit deinen PDF")

    # custom template for input & output
    user_question = st.text_input("Frage:", placeholder="Stelle hier deine Frage")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Deine Dokumente")

        # Upload PDF
        pdf_docs = st.file_uploader("Lade deine PDF's hier hoch und drücke auf 'Process'",
                                    accept_multiple_files=True,
                                    type=['pdf', 'txt'])

        if st.button("Process"):
            # If Button pressed
            with st.spinner("Loading"):

                # get raw text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                chunks = get_chunks(raw_text)

                # create vector store
                vector_store = create_vector_store(chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()