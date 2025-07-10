from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
    huggingfacehub_api_token=api_token,
    max_new_tokens=20,
    temperature=0,
    model_kwargs={"stop": ["\n"]}
)


model = ChatHuggingFace(llm=llm,)
result = model.invoke("What is the capital of Nepal?")
print(result.content)
