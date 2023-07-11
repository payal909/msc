import os
# from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from IPython.display import display, Markdown
import pandas as pd
import gradio as gr
import random
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma