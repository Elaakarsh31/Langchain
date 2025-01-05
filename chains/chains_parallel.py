from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# load the env variables 
load_dotenv()

# Instantiate the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define prompt templates  
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)

# Define analyze pros step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the pros of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)

# Define analyze pros step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features.")
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine pros and cons
def combine(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Create pros and cons Runnable branches
pros_chain_branch = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
cons_chain_branch = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()

# Create main chain
chain = (prompt_template 
         | model 
         | StrOutputParser() 
         | RunnableParallel(branches={"pros": pros_chain_branch, "cons": cons_chain_branch})
         | RunnableLambda(lambda x: print("final output", x) or combine(x["branches"]["pros"], x["branches"]["cons"]))
    )

# Run chain
result = chain.invoke({"product_name": "MacBook Pro"})

#Output
print(result)
