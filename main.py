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
import utils

utils.setup_page()

session = st.session_state
if 'transcript' not in session:
    session.transcript = []
if 'analysis' not in session:
    session.analysis = []
if 'input_disabled' not in session:
    session.input_disabled = True
if 'analyze_disabled' not in session:
    session.analyze_disabled = False
if 'institute' not in session:
    session.institute = ""
if 'institute_type' not in session:
    session.institute_type = "" 

embedding_llm, embeddings, chat_llm = utils.setup_llm()

institute_data_paths = {
    "Bank of Montreal (BMO)"            :   "./data/bmo_ar2022 (2).pdf",
    "Versa Bank"                        :   "./data/Versa bank",
    "National Bank of Canada (NBC)"     :   "./data/NATIONAL BANK OF CANADA_ 2022 Annual Report (1).pdf"
    }

with st.sidebar:
    institute = st.selectbox(label="Institute",options=institute_data_paths)
    session.institute = institute

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

def get_answer(question):
    # agent = RetrievalQA.from_chain_type(llm = embedding_llm,
    #     chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
    #     retriever=bank_db.as_retriever(),
    #     verbose=False,
    #     chain_type_kwargs={
    #     "verbose":True,
    #     "prompt": prompt,
    #     "memory": ConversationBufferMemory(
    #         input_key="question"),
    # })
    # return agent.run(question)
    return "Yes"

def analyse():
    with st.sidebar:
        with st.spinner(f"Checking if {session.institute} belongs to BCAR Short Form Category"):
            session.analyze_disabled = True
            session.analysis.append("The first step is to figure out whether the institute belong to BCAR Short Form, Category III or Full BCAR category.\n\nTo determine which of the above category the institute belongs to, you need to answer a series of questions.")
            q1_ans = get_answer(q1)
            session.analysis.append(f"1) {q1} {q1_ans}")
            session.institute_type = "Short Form"
            possibly_cat3 = False
            if q1_ans.startswith("Yes"):
                for qs in q1y_list:
                    qs_ans = get_answer(qs)
                    session.analysis.append(f"{2+q1y_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
            elif q1_ans.startswith("No"):
                for qs in q1n_list:
                    qs_ans = get_answer(qs)
                    session.analysis.append(f"{2+q1n_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
    with st.sidebar:
        with st.spinner(f"Checking if {session.institute} belongs to BCAR Category III Category"):
            if possibly_cat3:
                session.analysis.append("Based on the answers of the above question the institude does not come under BCAR Short Form Category. We will now check if it comes under BCAR Category III")
                session.institute_type = "Category 3"
                q2_ans = get_answer(q2)
                session.analysis.append(f"1) {q2} {q2_ans}")    
                if q2_ans.startswith("Yes"):
                    for qs in q2y_list:
                        qs_ans = get_answer(qs)
                        session.analysis.append(f"{2+q2y_list.index(qs)}) {qs} {qs_ans}")    
                        if qs_ans.startswith("Yes"):
                            session.analysis.append(f"Based on the answers of the above question {session.institute} does not come under BCAR Short Form or BCAR Category II so it belongs to Full BCAR Category")
                            session.institute_type = "Full Form"
                            break
                        session.analysis.append(f"Based on the answers of the above question {session.institute} comes under BCAR Category III")
                else:
                    session.analysis.append(f"Based on the answers of the above question {session.institute} does not come under BCAR Short Form or BCAR Category III so it belongs to Full BCAR Category")
                    session.institute_type = "Full Form"
            else:
                session.analysis.append(f"Based on the answers of the above question {session.institute} comes under BCAR Short Form Category")
    session.input_disabled = False

with st.sidebar:
    analyze_button = st.button("Analyze",use_container_width=True,disabled=session.analyze_disabled,on_click=analyse)
    for message in session.analysis:
        st.write(message)

docs = {
    f"{institute} Annual Report"                :   utils.load_doc(institute_data_paths[institute]),
    "Basel Capital Adequacy Reporting (BCAR)"   :   utils.load_doc("./data/Basel Capital Adequacy Reporting (BCAR) 2023 (2).pdf"),
    }

user_input = st.chat_input("Query",disabled=session.input_disabled)

if user_input:
    session.transcript.append(["user",user_input])
    response = utils.compare_answer(chat_llm,user_input,docs)
    session.transcript.append(["assistant",response])

if len(session.transcript)>0:
    for message in session.transcript:
        with st.chat_message(message[0]):
            st.write(message[1])

