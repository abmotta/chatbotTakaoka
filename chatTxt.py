from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
from streamlit_chat import message



#Carrega o PDF e salva o texto na variável text

def create_vectorstore():
    loader = TextLoader("./orientacoes.txt")
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(doc)
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets.openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    return vectorstore


if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = create_vectorstore()

retriever = st.session_state.vectorstore.as_retriever()

template = """Você é um médico que orienta suspensao pre-operatoria de medicacoes. Responda com base somente no 
seguinte contexto: {context}. Explique particularidades da medicação em questão. Não dê justificativa para informação 
que não encontrar no contexto. Não coloque uma frase de conclusão. Caso
nao encontre a medicacao, Responda: Desculpe, não tenho informação sobre esta medicação. 

Questão: Como fazer o manejo da {question} antes da cirurgia?
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(openai_api_key=st.secrets.openai_api_key)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

#st.markdown("<h1 style='text-align: center; color: black;'>⚕‍🤖Pergunte para o Taka 🩺️💊</h1>", unsafe_allow_html=True)

st.header("⚕‍🤖Pergunte para o Taka 🩺️💊")

if "msgs" not in st.session_state.keys():
    st.session_state.msgs = [{"is_user": False, "content": "Olá! Eu sou o Taka, o assistente virtual da Takaoka "
                                                           "Anestesia! Permita-me auxiliá-lo(a) no manejo perioperatório "
                                                           "de medicações.", "logo": logo_robot}]

def generate_response(user_question):
    ia_response = chain.invoke(user_question)
    st.session_state.msgs.append({"is_user": True, "content": user_question, "logo": logo_med})
    st.session_state.msgs.append({"is_user": False, "content": ia_response, "logo": logo_robot})

user_question = st.chat_input('Digite o nome da medicação')

if user_question:
    generate_response(user_question)

for msg in st.session_state.msgs:
    message(msg["content"], is_user=msg["is_user"], logo=msg["logo"])



