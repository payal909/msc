import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import OpenAI, VectorDBQA
from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

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
import utils

utils.setup_page()

session = utils.setup_session()

embedding_llm, embeddings, chat_llm = utils.setup_llm()

with st.sidebar:
    institute = st.selectbox(label="Institute",options=institute_names)
    session.institude = institude
    
institute_data_paths = {
    "Bank of Montreal (BMO)"            :   "./data/bmo_ar2022 (2).pdf",
    "Versa Bank"                        :   "./data/Versa bank",
    "National Bank of Canada (NBC)"     :   "./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf"
    }

docs = {
    f"{institute} Annual Report"                :   utils.load_doc(institute_data_paths[institute]),
    "Basel Capital Adequacy Reporting (BCAR)"   :   utils.load_doc("./data/Basel Capital Adequacy Reporting (BCAR) 2023 (2).pdf"),
    }

q1 = f"Does {session.institute} have a parent company?"
q1y_list = [
    f"Is {session.institute}'s parent an operating company regulated by OSFI?",
    f"Has {session.institute}'s parent adopted an internal rating (IRB) approach to credit risk?",
    f"Is {session.institute} a fully- consolidated subsidiary?",
    f"Does {session.institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q1n_list = [
    f"Has {session.institute} adopted an internal rating (IRB) approach to credit risk?",
    f"Is {session.institute} a fully- consolidated subsidiary?",
    f"Does {session.institute} have at least 95% of its credit risk exposures captured under the IRB approach?"
    ]
q2 = f"Is {session.institute} reporting less than $10 billion in total assets?"
q2y_list = [
    f"Is {session.institute} reporting greater than $100 million in total loans?",
    f"Does {session.institute} have an interest rate or foreign exchange derivatives with a combined notional amount greater than 100% of total capital?",
    f"Does {session.institute} have any other types of derivative exposure?",
    f"Does {session.institute} have exposure to other off-balance sheet items greater than 100% of total capital?"
    ]
questions = [q1,q1y_list,q1n_list,q2,q2y_list]

# with st.sidebar:
#     analyze_button = st.button("Analyze",use_container_width=True,disabled=session.analyze_disabled,on_click=utils.analyse)
#     for message in session.analysis:
#         st.write(message) 

if user_input:
    session.transcript.append(["user",user_input])
    response = utils.compare_answer(user_input,docs)
    session.transcript.append(["assistant",response])

if len(session.transcript)>0:
    for message in session.transcript:
        with st.chat_message(message[0]):
            st.write(message[1])

