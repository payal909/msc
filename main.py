import os
# from dotenv import load_dotenv, find_dotenv
# from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
# from langchain.llms import AzureOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
# from IPython.display import display, Markdown
# import pandas as pd
# import gradio as gr
# from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
import pandas as pd
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

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
import utils
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

utils.setup_page()

session = st.session_state
utils.setup_session(session)

embedding_llm, embeddings, chat_llm = utils.setup_llm()

all_documents = {
    "BCAR"                              :   {"data":"./data/Basel Capital Adequacy Reporting (BCAR) 2023 (2).pdf","index":"Basel Capital Adequacy Reporting (BCAR) 2023 (2)_index"},
    "Bank of Montreal (BMO)"            :   {"data":"./data/bmo_ar2022 (2).pdf","index":"bmo_ar2022 (2)_index"},
    "Versa Bank (VB)"                   :   {"data":"./data/Versa bank","index":"VBAR_index"},
    "National Bank of Canada (NBC)"     :   {"data":"./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf","index":"NATIONAL BANK OF CANADA_ 2022 Annual Report (1)_index"},
    }

institutes = all_documents.copy()
del institutes["BCAR"]

with st.sidebar:
    institute = st.selectbox(label="Institute",options=institutes)
    session.institute = institute

bank_db = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name=institutes[institute]["index"])

q1 = f"Does {institute} have a parent company?"
q1y_list = [
    f"Is {institute}'s parent an operating company regulated by OSFI?",
    f"Has {institute}'s parent adopted an internal rating (IRB) approach to credit risk?",
    f"Is {institute} a fully- consolidated subsidiary?",
    f"Does {institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q1n_list = [
    f"Has {institute} adopted an internal rating (IRB) approach to credit risk?",
    f"Is {institute} a fully- consolidated subsidiary?",
    f"Does {institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q2 = f"Is {institute} reporting less than $10 billion in total assets?"
q2y_list = [
    f"Is {institute} reporting greater than $100 million in total loans?",
    f"Does {institute} have an interest rate or foreign exchange derivatives with a combined notional amount greater than 100% of total capital?",
    f"Does {institute} have any other types of derivative exposure?",
    f"Does {institute} have exposure to other off-balance sheet items greater than 100% of total capital?"
    ]
    
questions = (q1,q1y_list,q1n_list,q2,q2y_list)

def analyse():
    utils.analyse(questions,session,embedding_llm,bank_db)

with st.sidebar:
    analyze_button = st.button("Analyze",use_container_width=True,disabled=session.analyze_disabled,on_click=analyse)
    for message in session.analysis:
        st.write(message)                           
        
docs = {
    f"{institute} Annual Report"                :   utils.load_doc(all_documents[institute]["data"]),
    "Basel Capital Adequacy Reporting (BCAR)"   :   utils.load_doc(all_documents["BCAR"]["data"]),
    "Analysis Report"                           :   session.analysis_text,
    }                                   

user_input = st.chat_input("Query",disabled=session.input_disabled)

if user_input:
    session.transcript.append(["user",user_input])
    bot_output = utils.compare_answer(chat_llm,session,user_input,docs)
    session.transcript.append(["assistant",bot_output])

if len(session.transcript)>0:
    for message in session.transcript:
        st.chat_message(message[0]).write(message[1])

