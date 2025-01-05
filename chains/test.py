from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
from langchain.schema.runnable import RunnableLambda

# load the env variables 
load_dotenv()

# Instantiate the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt templates  
prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)
prompt_template_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian judge who givs scores from 1 to 10"),
        ("human", "Score these jokes: {jokes}.")
    ]
)

# chain_1 = LLMChain(llm=model, prompt= prompt_template_1, output_key="jokes")
# chain_2 = LLMChain(llm=model, prompt= prompt_template_2, output_key="scores")
# # Create the combined chain with Langchain Expression Language (LCEL)
chain_1 = prompt_template_1 | model | StrOutputParser()
chain_2 = prompt_template_2 | model | StrOutputParser()

display_jokes = RunnableLambda(lambda x: (print(f"Jokes: {x}"), x)[1])

chain = chain_1 | display_jokes | chain_2

# result = chain.invoke({'topic'})

# chain = SequentialChain(
#     chains = [chain_1, chain_2],
#     input_variables= ["topic", "joke_count"],
#     output_variables=["jokes", "scores"]
# )

result = chain.invoke({'topic': 'lawyers', 'joke_count':2})

print(result)
