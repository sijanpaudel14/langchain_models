from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# os.environ['HF_HOME'] = "D:/huggingfae_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="distilbert/distilgpt2",
    task='text-completion',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Nepal?")
print(result.content)
