from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
from langchain.chat_models import init_chat_model
model = init_chat_model("llama3-8b-8192", model_provider="groq")
model.invoke("Explain me about neural networks")
from langchain_core.messages import HumanMessage, SystemMessage
model.invoke("Hello")
model.invoke([{"role": "user", "content": "Hello"}])
res=model.invoke([HumanMessage("Hello")])
res.content
from langchain_core.prompts import ChatPromptTemplate
travel_temp="suggest me name of a place in one word to travel in {country}"
prompt_template=ChatPromptTemplate=ChatPromptTemplate.from_messages([
    ("system",travel_temp),
     ("user","{country}")
])
