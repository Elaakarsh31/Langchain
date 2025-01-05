from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# load the env
load_dotenv()

# model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Create chat prompt template
print("-----Prompt from Template-----\n")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)

# Part 2: with multiple variables
print("\n------ Multiple variables ------\n")
messages = [
        ("system", "You are a comedian who tells a joke about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 2})
result = model.invoke(prompt)
print(result.content)