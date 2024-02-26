import config
import tiktoken
from openai import AsyncAzureOpenAI 
import openai
import json 
import numpy as np

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
client = AsyncAzureOpenAI(azure_endpoint = config.openai_api_base, api_key= config.openai_api_key, api_version='2023-05-15')

class ChatGPT:
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in {"gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"}, f"Unknown model: {model}"
        self.model = model
        self.map_dict = {"gpt-3.5-turbo": "gpt-35-turbo-16k", "gpt-3.5-turbo-16k": "gpt-35-turbo-16k", "gpt-4": "gpt-4"}
    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await client.chat.completions.create(
                        model=self.map_dict.get(self.model, self.model),
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message.content
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.BadRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, image_url = None, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode, image_url)
                    client = AsyncAzureOpenAI(azure_endpoint = config.openai_api_base, api_key= config.openai_api_key, api_version='2023-05-15')
                    if image_url:
                        client = AsyncAzureOpenAI(azure_endpoint = config.openai_v_api_base, api_key= config.openai_v_api_key, api_version='2023-05-15')
                    r_gen = await client.chat.completions.create(
                        model='gpt-4v' if image_url else self.map_dict.get(self.model, self.model),
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = ""
                    async for r_item in r_gen:
                        if len(r_item.choices) == 0:
                            raise ValueError(f"choices is empty: {r_item}, input message: {messages}")
                        delta = r_item.choices[0].delta
                        if delta.content:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                answer = self._postprocess_answer(answer)

            except openai.BadRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode, image_url=None):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}] if prompt else []
        if image_url:
            messages = []
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        if image_url:
            content = [{"type": "text", "text": message}]
            content.append({"type": "image_url", "image_url": {"url": image_url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": message})
        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-4-1106-preview":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, list):
                    n_input_tokens += len(encoding.encode(value[0]['text']))
                else:
                    n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file) -> str:
    client = AsyncAzureOpenAI(azure_endpoint = config.openai_api_base, api_key= config.openai_api_key, api_version='2023-05-15')
    r = await client.audio.transcriptions.create(("whisper-1", audio_file))
    return r["text"] or ""


async def generate_images(prompt, n_images=4, size="512x512"):
    client = AsyncAzureOpenAI(azure_endpoint = config.dalle_api_base, api_key= config.dalle_api_key, api_version="2023-12-01-preview")
    r = await client.images.generate(model='dalle3', prompt=prompt, n=n_images, size=size)
    json_response = json.loads(r.model_dump_json())
    print(json_response)
    image_urls = [item["url"] for item in json_response["data"]]
    return image_urls

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

async def get_embedding(text: str):
    # Call the OpenAI API to get the embedding
    client = AsyncAzureOpenAI(azure_endpoint = config.openai_api_base, api_key= config.openai_api_key, api_version='2023-05-15')
    response = await client.embeddings.create(input = text, model= "text-embedding-ada-002")
    cut_dim = response.data[0].embedding[:config.embedding_dim]
    norm_dim = normalize_l2(cut_dim).tolist()
    return norm_dim

