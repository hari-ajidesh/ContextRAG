import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Sidebar setup
with st.sidebar:
   st.header("PDF Document Setup")
   uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

   # Check if the setup is done or not
   if "setup_done" not in st.session_state:
       # If no file is uploaded, display a message
       if uploaded_file is None:
           st.markdown("Waiting for input file")
       else:
           # Save the uploaded PDF file temporarily
           with open("temp.pdf", "wb") as f:
               f.write(uploaded_file.getbuffer())
           
           # Load the PDF
           start_time = time.time()
           loader = PyPDFLoader("temp.pdf")
           st.session_state.docs = loader.load()

           st.info(f"⏱ Data loaded in {time.time() - start_time:.2f} seconds")
           time1 = time.time()

           # Split the documents into smaller chunks
           text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
           st.session_state.splits = text_splitter.split_documents(st.session_state.docs)

           st.info(f"⏱ Text split in {time.time() - time1:.2f} seconds")
           time2 = time.time()

           # Create embeddings and vector store
           st.session_state.vectorstore = Chroma.from_documents(documents=st.session_state.splits, embedding=OllamaEmbeddings())
           st.session_state.retriever = st.session_state.vectorstore.as_retriever()

           st.info(f"⏱ Vector database formed in {time.time() - time2:.2f} seconds")
           time3 = time.time()

           # Initialize LLM and prompts
           st.session_state.llm = Ollama(model="llama2")

           st.session_state.contextualize_q_system_prompt = """Given a chat history and the latest user question \
           which might reference context in the chat history, formulate a standalone question \
           which can be understood without the chat history. Do NOT answer the question, \
           just reformulate it if needed and otherwise return it as is."""
           st.session_state.contextualize_q_prompt = ChatPromptTemplate.from_messages(
               [
                   ("system", st.session_state.contextualize_q_system_prompt),
                   MessagesPlaceholder("chat_history"),
                   ("human", "{input}"),
               ]
           )

           st.session_state.history_aware_retriever = create_history_aware_retriever(
               st.session_state.llm, st.session_state.retriever, st.session_state.contextualize_q_prompt
           )

           st.session_state.qa_system_prompt = """You are an assistant for question-answering tasks. \
           Use the following pieces of retrieved context to answer the question. \
           If you don't know the answer, just say that you don't know. \
           Use three sentences maximum and keep the answer concise.\

           {context}"""
           st.session_state.qa_prompt = ChatPromptTemplate.from_messages(
               [
                   ("system", st.session_state.qa_system_prompt),
                   MessagesPlaceholder("chat_history"),
                   ("human", "{input}"),
               ]
           )
           st.session_state.question_answer_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.qa_prompt)

           st.session_state.rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, st.session_state.question_answer_chain)

           def get_session_history(session_id: str) -> BaseChatMessageHistory:
               # Initialize a new chat history if it doesn't exist
               if session_id not in store:
                   store[session_id] = ChatMessageHistory()
               return store[session_id]

           store = {}

           st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
               st.session_state.rag_chain,
               get_session_history,
               input_messages_key="input",
               history_messages_key="chat_history",
               output_messages_key="answer",
           )

           st.info(f"⏱ RAG Chain formed in {time.time() - time3:.2f} seconds")
           st.session_state.setup_done = True

# Chat interaction logic
if "messages" not in st.session_state:
   st.session_state.messages = []

# Display the previous messages in the chat
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

# Get user input and generate a response
if prompt := st.chat_input("How could I help you?"):
   time4 = time.time()
   # Add user message to chat history
   st.session_state.messages.append({"role": "user", "content": prompt})

   # Display user message in chat message container
   with st.chat_message("user"):
       st.markdown(prompt)

   with st.chat_message("assistant"):
       time6 = time.time()
       # Generate a response using the conversational RAG chain
       response = st.session_state.conversational_rag_chain.invoke(
           {"input": prompt}, config={"configurable": {"session_id": "abc123"}}
       )["answer"]

       st.info(f"⏱ Response generated in {time.time() - time6:.2f} seconds")
       time7 = time.time()

       st.markdown(response)

   # Add the assistant's response to the chat history
   st.session_state.messages.append({"role": "assistant", "content": response})