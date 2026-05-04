from openai import OpenAI
import json

class LLMSarcasmTranslator:

    SYSTEM_PROMPT = (
                    "You are a helpful assistant that rewrites sarcastic Slovenian sentences into their literal non-sarcastic meaning."
                    "Preserve language (including Slovenian)."
                    "IMPORTANT! Return just the output, without any additional commentary, explanation or notes."                                    
                )
    
    BASE_USER_PROMPT = "Here is some text: '{}'. Here is a rewrite of the text, which is not sarcastic:"
    
    def __init__(
        self,
        api_key,
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1"
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _call(self, messages, temperature=0.7, max_tokens=300):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def zero_shot(self, text):
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": self.BASE_USER_PROMPT.format(text)
            }
        ]
        return self._call(messages)

    def few_shot(self, text, examples_path="./llm_sarcasm_examples.json"):  
        
        examples = []
        with open(examples_path) as f:
            examples = json.load(f)
            
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]

        for ex in examples:

            messages.append({
                "role": "user",
                "content": f"Here is some text: {ex['ironic']}"
            })
            messages.append({
                "role": "assistant",
                "content": f"Here is a rewrite of the text, which is not sarcastic: {ex['literal']}"
            })

        messages.append({
            "role": "user",
            "content": self.BASE_USER_PROMPT.format(text)
        })

        return self._call(messages)

    '''def augmented_zero_shot(self, text, examples):
        """
        examples format:
        [{"ironic": "...", "literal": "...", "style": "..."}]
        style is optional
        """

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]

        for ex in examples:
            style = ex.get("style", "neutral")

            messages.append({
                "role": "user",
                "content": f"Rewrite ({style}): {ex['ironic']}"
            })
            messages.append({
                "role": "assistant",
                "content": ex["literal"]
            })

        messages.append({
            "role": "user",
            "content": f"Rewrite literally: {text}"
        })

        return self._call(messages)'''


    def self_consistency(self, text, n=5):
        """
        Generates multiple outputs and returns them all.
        You can later vote/aggregate.
        """
        results = []

        for _ in range(n):
            res = self.zero_shot(text)
            results.append(res)

        return results