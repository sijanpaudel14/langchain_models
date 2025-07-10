from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model="gpt-4")
# model = ChatOpenAI(model="gpt-4", temperature=1.8)
model = ChatOpenAI(model="gpt-4", temperature=1.8, max_completion_tokens=10)
result = model.invoke("What is the capital of Nepal?")
print(result.content)


# temperature
# 0.0 - 0.3 - more deterministic, less creative, Factual answers(math, code)
# 0.4 - 0.7 - balanced, good for most tasks(general QA, explanations)
# 0.8 - 1.2 - more creative, less factual, good for creative, storytelling, jokes
# 1.3 - 2.0 - very creative, less factual, good for brainstorming, maximum randomness, wild ideas

# tokens
# Roughly knows as words
