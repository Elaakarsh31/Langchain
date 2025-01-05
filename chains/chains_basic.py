from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# load the env variables 
load_dotenv()

# Instantiate the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt templates  
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain with Langchain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model 

# run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)