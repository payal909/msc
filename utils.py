import os
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
# from langchain.agents.tools import Tool
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
import pandas as pd
from tqdm import tqdm
# from langchain.document_loaders import UnstructuredPDFLoader

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def setup_page():
    st.set_page_config(layout="wide")

    hide = '''
    <style>
    MainMenu {visibility:hidden;}
    header {visibility:hidden;}
    footer {visibility:hidden;}
    .css-1b9x38r {
        display: none;
        }
    .css-1544g2n {
        padding: 1rem 1rem 1.5rem;
    }
    .css-1oy2v5l {
        margin-top: 0.7rem;
    }
    div.block-container {
        padding-top: 0rem;
        }
    </style>
    '''
    st.markdown(hide, unsafe_allow_html=True)
              
def setup_session(session):
    if 'transcript' not in session:
        session.transcript = []
    # if 'analysis' not in session:
    #     session.analysis = []
    if 'input_disabled' not in session:
        session.input_disabled = True
    if 'analyze_disabled' not in session:
        session.analyze_disabled = False
    if 'institute' not in session:
        session.institute = ""
    # if 'institute_type' not in session:
    #     session.institute_type = ""
    # if "analysis_text" not in session:
    #     session.analysis_text = ""

def setup_llm():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    # embedding_llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)
    
    os.environ["OPENAI_API_TYPE"] ="azure"
    os.environ["OPENAI_API_VERSION"] ="2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://testavinx.openai.azure.com/"

    openai_llm = AzureChatOpenAI(deployment_name="gpt-35-turbo",model_name="gpt-35-turbo",temperature=0)
    embeddings = OpenAIEmbeddings(deployment="embedding1",model="text-embedding-ada-002",openai_api_base="https://testavinx.openai.azure.com/",openai_api_type="azure",chunk_size = 1)
    
    

    claude_models = ["claude-instant-1","claude-2"]
    chat_llm = ChatAnthropic(model=claude_models[1],temperature= 0,max_tokens_to_sample = 512)
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    anthropic_llm = ChatAnthropic(model=claude_models[1],temperature= 0,max_tokens_to_sample = 512)

    
    return openai_llm, embeddings, anthropic_llm

def load_doc(path):
    k=300000
    if path.endswith(".pdf"):
        doc = PyPDFLoader(file_path=path)
    else:
        doc = DirectoryLoader(path=path,glob="**/*.pdf")
    document = doc.load()
    context = "\n\n".join([document[i].page_content for i in range(len(document))])
    return context[:k]

def compare_answer(summary_llm,chat_llm,session,question,docs):
    
    retrival_system_template = """You are a helpful assistant, You need to extract as much text as you can which is relater or relevant to the answer of the user question from the context provided.
Do not try to answer the question, just extract the text relevant to the answer of the user question.
Use the following context (delimited by <ctx></ctx>) for finding out the relevant text:

<ctx>
{context}
</ctx>"""
    
    retrival_system_prompt = SystemMessagePromptTemplate.from_template(template=retrival_system_template)
    messages = [retrival_system_prompt,HumanMessage(content=question)]
    compare_chat_prompt = ChatPromptTemplate.from_messages(messages)
    
    summary = dict()
    for doc_name,doc_txt in tqdm(docs.items()):
        summary[doc_name] = summary_llm(compare_chat_prompt.format_prompt(context=doc_txt).to_messages()).content

    compare_context = "\n\n".join([f"Relevant points from {doc_name}:\n\n{doc_summary}" for doc_name,doc_summary in summary.items()])
    
    compare_system_template = """You are a helpful chatbot who has to answer question of a user from the institute {institute}.
You will be given relevant points from various documents that will help you answer the user question.
Below is a list of relevant points along with the name of the document from where thoes points are from.
Consider all the documents provided to you and answer the question by choosing all the relevant points to the question.
You might have to compare points from more than one document to answer the question.

{context}"""

    compare_system_prompt = SystemMessagePromptTemplate.from_template(template=compare_system_template)
    messages = [compare_system_prompt,HumanMessage(content=question)]
    compare_chat_prompt = ChatPromptTemplate.from_messages(messages)
    response = chat_llm(compare_chat_prompt.format_prompt(institute=session.institute,question=question,context=compare_context).to_messages()).content
    return response
