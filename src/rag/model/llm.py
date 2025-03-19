from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        set_seed(42)
        self.pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
            device=self.device,
        )

    @abstractmethod
    def get_prompt(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        return self.pipe(prompt)[0]["generated_text"]


class AnswerLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prompt(self, query_text: str, context_text: str):
        PROMPT_FORMAT = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        If applicable, provide the source document number and include relevant metadata (e.g., timestamp, source, or other details).
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}
        """,
            },
        ]
        prompt_template = self.tokenizer.apply_chat_template(
            PROMPT_FORMAT, tokenize=False, add_generation_prompt=True
        )
        return prompt_template.format(question=query_text, context=context_text)
