from openai import OpenAI
from .base import LanguageModel

# 本地 Ollama 的默认地址
QWEN_ENDPOINT = "http://localhost:11434/v1"
QWEN_API_KEY = "ollama-not-needed"  # 不需要真正的 key

class QwenServer(LanguageModel):
    def __init__(self, model: str = "qwen2.5:7b", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.client = OpenAI(
            base_url=QWEN_ENDPOINT,
            api_key=QWEN_API_KEY,
        )

    def chat(self, message: str, system_msg: str = None, json_mode: bool = False):
        if system_msg is None:
            system_msg = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=60
            )
            result = response.choices[0].message.content
            print(f"🔍 Raw LLM response:\n{result}\n{'='*40}")
            return result or "[EMPTY]"
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return "[ERROR]"

    def complete(self, prompts: str):
        response = self.client.completions.create(
            model=self.model, prompt=prompts, echo=False, max_tokens=100
        )
        return response.choices[0].text


if __name__ == "__main__":
    qwen = QwenServer("qwen2.5:7b")
    response = qwen.complete("The reason of human landing on moon is that, someone found it strange behind the moon.")
    print(response)