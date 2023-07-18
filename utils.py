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
        
    .css-1cypcdb {
        min-width: 500px;
        max-width: 500px;
    }
    .css-1544g2n {
        padding: 1rem 1rem 1.5rem;
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

def setup_llm():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    embedding_llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size =1)
    
    claude_models = ["claude-instant-1","claude-2"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    chat_llm = ChatAnthropic(model=claude_models[1],temperature= 0)
    
    return embedding_llm, embeddings, chat_llm

def load_doc(path):
    if path.endswith(".pdf"):
        doc = PyPDFLoader(file_path=path)
    else:
        doc = DirectoryLoader(path=path,glob="**/*.pdf")
    document = doc.load()
    context = "\n\n".join([document[i].page_content for i in range(len(document))])
    return context[:300000]

def get_answer(llm,db,question):
    template = """
You are a helpful virtual assistant of OSFI. Analyze the context and answer the question in one word "Yes" or "No" only. Remember the
answer should be only "Yes" or "No". If you don't know the answer, just answer "No".
Use the following context (delimited by <ctx></ctx>) to answer the question:

------
<ctx>
{context}
</ctx>
------
{question}
Answer:
"""

    prompt = PromptTemplate(input_variables=["context", "question"],template=template)
    agent = RetrievalQA.from_chain_type(llm = llm,
        chain_type='stuff', # 'stuff', 'map_reduce', 'refine', 'map_rerank'
        retriever=db.as_retriever(),
        verbose=False,
        chain_type_kwargs={
        "verbose":False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            input_key="question"),
    })
    return agent.run(question)

def analyse(questions,session,llm,db):
    q1,q1y_list,q1n_list,q2,q2y_list = questions
    with st.sidebar:
        with st.spinner(f"Checking if {session.institute} belongs to BCAR Short Form Category"):
            session.analyze_disabled = True
            session.analysis.append(f"The first step is to figure out whether {session.institute} belong to BCAR Short Form, Category III or Full BCAR category.\n\nTo determine which of the above category the institute belongs to, the following series of questions need to be answered.")
            q1_ans = get_answer(llm,db,q1)
            session.analysis.append(f"1) {q1} {q1_ans}")
            session.institute_type = "Short Form"
            possibly_cat3 = False
            if q1_ans.startswith("Yes"):
                for qs in q1y_list:
                    qs_ans = get_answer(llm,db,qs)
                    session.analysis.append(f"{2+q1y_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
            elif q1_ans.startswith("No"):
                for qs in q1n_list:
                    qs_ans = get_answer(llm,db,qs)
                    session.analysis.append(f"{2+q1n_list.index(qs)}) {qs} {qs_ans}")    
                    if qs_ans.startswith("No"):
                        possibly_cat3 = True
                        break
    with st.sidebar:
        with st.spinner(f"Checking if {session.institute} belongs to BCAR Category III Category"):
            if possibly_cat3:
                session.analysis.append(f"Based on the answers of the above questions {session.institute} does not come under BCAR Short Form Category. To determine if it comes under BCAR Category III the following series of questions need to be answered.")
                session.institute_type = "Category 3"
                q2_ans = get_answer(llm,db,q2)
                session.analysis.append(f"1) {q2} {q2_ans}")    
                if q2_ans.startswith("Yes"):
                    for qs in q2y_list:
                        qs_ans = get_answer(llm,db,qs)
                        session.analysis.append(f"{2+q2y_list.index(qs)}) {qs} {qs_ans}")    
                        if qs_ans.startswith("Yes"):
                            session.analysis.append(f"Based on the answers of the above questions {session.institute} does not come under BCAR Short Form or BCAR Category III so it belongs to Full BCAR Category")
                            session.institute_type = "Full Form"
                            break
                        session.analysis.append(f"Based on the answers of the above questions {session.institute} comes under BCAR Category III")
                else:
                    session.analysis.append(f"Based on the answers of the above questions {session.institute} does not come under BCAR Short Form or BCAR Category III so it belongs to Full BCAR Category")
                    session.institute_type = "Full Form"
            else:
                session.analysis.append(f"Based on the answers of the above questions {session.institute} comes under BCAR Short Form Category")
            session.input_disabled = False
            
    schedules = pd.read_csv("schedules.csv",delimiter="|")
    limited_schedules = schedules[schedules[session.institute_type]][["Schedule Number","Schedules"]]
    # limited_schedules = "\n".join([f"{i+1}) {limited_schedules[i]}\n" for i in range(len(limited_schedules))])
    session.transcript.append(f"According to the information provided {session.institute} belongs to {session.institute_type} category and thus the required schedules are limited to:")
    session.transcript.append(limited_schedules)
    
    # analysis_text = "\n\n".join(session.analysis)+"\n\n"+session.transcript[0]+"\n\n"+session.transcript[0].to_markdown()
    # with st.expander("analysis"):
    #     st.write(analysis_text)
    # analysis_db = FAISS.from_texts([analysis_text],embeddings)

def compare_answer(chat_llm,question,docs):
    
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
        summary[doc_name] = chat_llm(compare_chat_prompt.format_prompt(context=doc_txt).to_messages()).content

    compare_context = "\n\n".join([f"Relevant points from {doc_name}:\n\n{doc_summary}" for doc_name,doc_summary in summary.items()])
    
    compare_system_template = """You are a helpful chatbot who has to answer question of a user from the institute {institute} which comes under the BCAR {institute_type} section.
You will be given relevant points from various documents that will help you answer the user question.
Below is a list of relevant points along with the name of the document from where thoes points are from.
Consider all the documents provided to you and answer the question by choosing all the relevant points to the question.
You might have to compare points from more than one document to answer the question.

{context}"""

    compare_system_prompt = SystemMessagePromptTemplate.from_template(template=compare_system_template)
    messages = [compare_system_prompt,HumanMessage(content=question)]
    compare_chat_prompt = ChatPromptTemplate.from_messages(messages)
    response = chat_llm(compare_chat_prompt.format_prompt(institute=institute,institute_type=institute_type,question=question,context=compare_context).to_messages()).content
    return response

# def get_answer(question):
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
    # pass

# def analyse(questions,session):
        # q1,q1y_list,q1n_list,q2,q2y_list = questions
        # with st.sidebar:
        #     with st.spinner(f"Checking if {session.institute} belongs to BCAR Short Form Category"):
        #         session.analyze_disabled = True
        #         session.analysis.append("The first step is to figure out whether the institute belong to BCAR Short Form, Category III or Full BCAR category.\n\nTo determine which of the above category the institute belongs to, you need to answer a series of questions.")
        #         q1_ans = get_answer(q1)
        #         session.analysis.append(f"1) {q1} {q1_ans}")
        #         session.institute_type = "Short Form"
        #         possibly_cat3 = False
        #         if q1_ans.startswith("Yes"):
        #             for qs in q1y_list:
        #                 qs_ans = get_answer(qs)
        #                 session.analysis.append(f"{2+q1y_list.index(qs)}) {qs} {qs_ans}")    
        #                 if qs_ans.startswith("No"):
        #                     possibly_cat3 = True
        #                     break
        #         elif q1_ans.startswith("No"):
        #             for qs in q1n_list:
        #                 qs_ans = get_answer(qs)
        #                 session.analysis.append(f"{2+q1n_list.index(qs)}) {qs} {qs_ans}")    
        #                 if qs_ans.startswith("No"):
        #                     possibly_cat3 = True
        #                     break
        # with st.sidebar:
        #     with st.spinner(f"Checking if {session.institute} belongs to BCAR Category III Category"):
        #         if possibly_cat3:
        #             session.analysis.append("Based on the answers of the above question the institude does not come under BCAR Short Form Category. We will now check if it comes under BCAR Category III")
        #             session.institute_type = "Category 3"
        #             q2_ans = get_answer(q2)
        #             session.analysis.append(f"1) {q2} {q2_ans}")    
        #             if q2_ans.startswith("Yes"):
        #                 for qs in q2y_list:
        #                     qs_ans = get_answer(qs)
        #                     session.analysis.append(f"{2+q2y_list.index(qs)}) {qs} {qs_ans}")    
        #                     if qs_ans.startswith("Yes"):
        #                         session.analysis.append(f"Based on the answers of the above question {session.institute} does not come under BCAR Short Form or BCAR Category II so it belongs to Full BCAR Category")
        #                         session.institute_type = "Full Form"
        #                         break
        #                     session.analysis.append(f"Based on the answers of the above question {session.institute} comes under BCAR Category III")
        #             else:
        #                 session.analysis.append(f"Based on the answers of the above question {session.institute} does not come under BCAR Short Form or BCAR Category III so it belongs to Full BCAR Category")
        #                 session.institute_type = "Full Form"
        #         else:
        #             session.analysis.append(f"Based on the answers of the above question {session.institute} comes under BCAR Short Form Category")
        # session.input_disabled = False

    # schedules = pd.read_csv("schedules.csv",delimiter="|")
    # limited_schedules = schedules[schedules[session.institute_type]][["Schedule Number","Schedules"]]
    # # limited_schedules = "\n".join([f"{i+1}) {limited_schedules[i]}\n" for i in range(len(limited_schedules))])
    # session.transcript.append(f"According to the information provided the Institute belongs to {session.institute_type} category and thus the required schedules are limited to:")
    # session.transcript.append(limited_schedules)                                                           

# analysis_text = "\n\n".join(session.analysis)+"\n\n"+session.transcript[0]+"\n\n"+session.transcript[0].to_markdown()
# with st.expander("analysis"):
#     st.write(analysis_text)
# analysis_db = FAISS.from_texts([analysis_text],embeddings)

