from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI

# load the env variables 
load_dotenv()

# Instantiate the model
# model = OpenAI()
model = ChatOpenAI(model="gpt-3.5-turbo")
llm = OpenAI()
# Define the input as a single prompt
input_text = "What is your name?"

# Invoke the model
# result = model.invoke(input_text)

# Print the Output
print(f"ChatOpenAI: {model.invoke(input_text).content}\nOpenAI: {llm.invoke(input_text)}")