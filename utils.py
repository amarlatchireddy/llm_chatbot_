import configparser
import os
from langchain_community.utilities import SQLDatabase

from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


def read_properties_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The mention file '{path}' is not found")
    
    config = configparser.ConfigParser()
    config.read(path)
    db_path = config['default']["db"]
    gemini_api_key = config['default']['google_gemini_key']

    return db_path, gemini_api_key


def get_properties():
    
    path = 'config.properties'

    try:
        db_path,gen_Ai_key = read_properties_file(path)
        return db_path,gen_Ai_key

    except FileNotFoundError as e:
        print(e)

def get_llm(gemini_api_key):
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, 
                                 convert_system_message_to_human=True, temperature=0.0)
    return llm

def db_connection(db):
    db = SQLDatabase.from_uri(f"sqlite:///{db}")
    print(db.dialect)
    print(db.get_usable_table_names())
    resp = db.run("SELECT * FROM Employees LIMIT 10;")
    print(resp)
    return db

def create_conversational_chain():

    try:
        db, gemini_api_key = get_properties()

        # Get the instance of LLM
        llm = get_llm(gemini_api_key)
        # Get the DB connection
        db = db_connection(db)

        sql_prompt_template = """
        Only use the following tables:
        {table_info}
        Question: {input}

        Given an input question, first create a syntactically correct
        {dialect} query to run.
        
        Relevant pieces of previous conversation:
        {history}

        (You do not need to use these pieces of information if not relevant)
        Dont include ```, ```sql and \n in the output.
        """
        prompt = PromptTemplate(
                input_variables=["input", "table_info", "dialect", "history"],
                template=sql_prompt_template,
            )
        memory = ConversationBufferMemory(memory_key="history")

        
        db_chain = SQLDatabaseChain.from_llm(
                llm, db, memory=memory, prompt=prompt, return_direct=True,  verbose=True
            )

        output_parser = StrOutputParser()
        chain = llm | output_parser
        

    except Exception as e:
        raise e
    return  db_chain, chain

