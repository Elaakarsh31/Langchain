from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

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

# Define Additional Processing steps
upper_case = RunnableLambda(lambda x: x.upper())
word_count = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create combined chain
chain = prompt_template | model | StrOutputParser() | upper_case | word_count

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 1})

print(result)