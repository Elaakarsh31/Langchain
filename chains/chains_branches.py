from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

# load the env variables 
load_dotenv()

# Instantiate the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Positive Template
positive_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", "Give a thank you not for this feedback: {feedback}.")
    ]
)

# Negative feedback
negative_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}.")
])

# Neutral feedback
neutral_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}.")
])

# Escalate feedback
escalate_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}.")
])

# Classification Prompt
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant."),
    ("human", "Classifiy the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}.")
])

# Branch
branch = RunnableBranch(
    (
        lambda x: "positive" in x, positive_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x, negative_template | model | StrOutputParser(),
    ),
    (
        lambda x: "neutral" in x, neutral_template | model | StrOutputParser()
    ),
    escalate_template | model | StrOutputParser()
)

# Create the classificatoin sub chain
classification_chain = classification_prompt | model | StrOutputParser()

# Create the main chain
chain = classification_chain | branch

# Sample review
review = "The product is excellent, I really enjoyed using and found it very helpful."

result = chain.invoke({"feedback": review})

print(result)