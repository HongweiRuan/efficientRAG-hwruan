from src.language_models.llama import LlamaServer

llm = LlamaServer("qwen2.5:7b")
res = llm.chat("What is 2 + 2?")
print(f"✅ Response: {res}")