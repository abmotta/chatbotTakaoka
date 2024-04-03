from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import NotionDBLoader

NOTION_TOKEN = st.secrets.notion_token
DATABASE_ID = st.secrets.database_id

def create_vectorstore():
    loader = NotionDBLoader(
        integration_token=NOTION_TOKEN,
        database_id=DATABASE_ID,
        request_timeout_sec=30,  # optional, defaults to 10
    )
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
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

template = """
Você é uma IA que orienta médicos na suspensão de medicamentos antes de cirurgias.
Responda de maneira objetiva, com base somente no seguinte contexto: {context}.
A título de informaçao, os nomes comerciais das medicações encontram-se seguidos pelo caractere especial ®.
Responda seguindo o exemplo: A medicação metformina, comercializada com o nome comercial Glifage®,
um antidiabético da classe das Biguanidas, deve ser suspensa apenas no dia do procedimento, 
segundo a seguinte referência: XXXXXX.
Por favor, nao confunda a orientação de medicaçoes com nomes parecidos, caso sejam divergentes.
Caso encontre informação adicional, informe.
Caso não encontre informação adicional, não mencione que não encontrou.
Não coloque uma frase de conclusao na resposta, para ser mais conciso.
Caso nao encontre a medicação, responda: 'Desculpe, não tenho informação sobre esta medicação'.

Question: Explique o manejo da {question} no período perioperatorio?
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(openai_api_key=st.secrets.openai_api_key, temperature=0.3)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

logo_robot = "https://raw.githubusercontent.com/abmotta/chatbotTakaoka/main/taka_robot.png"
logo_med = "https://raw.githubusercontent.com/abmotta/chatbotTakaoka/main/doctor_avatar_medical_icon_140443.png"

with st.container(height=80, border=False):
    st.header("⚕‍🤖Pergunte para o Taka 🩺️💊")


if "msgs" not in st.session_state.keys():
    st.session_state.msgs = [{"is_user": False, "content": "Olá! Eu sou o Taka, o assistente virtual da Takaoka "
                                                           "Anestesia! Permita-me auxiliá-lo(a) no manejo perioperatório "
                                                           "de medicações.", "logo": logo_robot}]

if "msg_id" not in st.session_state.keys():
    st.session_state.msg_id = 0
    
def gen_msg_id():
    st.session_state.msg_id += 1
    return st.session_state.msg_id

def generate_response(user_question):
    ia_response = chain.invoke(user_question)
    st.session_state.msgs.append({"is_user": True, "content": user_question, "logo": logo_med})
    st.session_state.msgs.append({"is_user": False, "content": ia_response, "logo": logo_robot})

user_question = st.chat_input('Digite o nome da medicação')

if user_question:
    generate_response(user_question)

for msg in st.session_state.msgs:
    key = str(gen_msg_id())
    message(msg["content"], is_user=msg["is_user"], logo=msg["logo"], key=key)


