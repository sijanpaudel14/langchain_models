from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
documents = [
    "Kathmandu is the capital of Nepal",
    'Paris is the capital of France',
    'New Delhi is the capital of India'
]

result = embedding.embed_query("Kathmandu is the capital of Nepal")

print(str(result))
