import base64
from typing import List
from dataclasses import dataclass
from io import BytesIO

from urllib.request import urlopen
import librosa

from architecture_registry_module.classes.model_entity import ModelEntity


@dataclass
class AudioTextToTextModelEntity(ModelEntity):
    def __call__(self, *args, **kwargs):
        return self.run_inference(*args, **kwargs)

    def run_inference(self, *args, **kwargs):
        first_arg = args[0]

        if isinstance(first_arg, str):
            # this is the standard input, which is text and audios
            return self.run_inference_default(*args, **kwargs)

        # supports chat messages
        first_arg: list
        return self.run_inference_chat(*args, **kwargs)

    def run_inference_default(self, text, audios, **kwargs):
        output = self.generate(text, audios, **kwargs)

        return output

    def run_inference_chat(self, *args, **kwargs):
        messages = args[0]

        adapted_messages, audios = self.adapt_to_conversational_chat_json(
            messages=messages
        )

        text = self.processor.apply_chat_template(
            adapted_messages, add_generation_prompt=True, tokenize=False
        )

        output = self.generate(text, audios, **kwargs)[0]

        output_messages = messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output["generated_text"]}],
            }
        ]

        return [{**output, "generated_text": output_messages}]

    def generate(self, text, audios, **kwargs):
        kwargs = {**kwargs, "streamer": kwargs.get("streamer")}

        audios = self.load_audios(audios)

        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        # move the tensors to the same device as the model
        inputs.input_ids = inputs.input_ids.to(self.model.device)

        generate_ids = self.model.generate(**inputs, **kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    def load_audios(self, audios):
        loaded_audios = []

        for audio_source in audios:
            # standard urls
            if audio_source.startswith("http"):
                audio_bytes = BytesIO(urlopen(audio_source).read())
            # base64
            else:
                if "," in audio_source:
                    b64_string = audio_source.split(",", 1)[1]
                else:
                    b64_string = audio_source
                audio_data = base64.b64decode(b64_string)
                audio_bytes = BytesIO(audio_data)

            loaded_audio, _ = librosa.load(
                audio_bytes, sr=self.processor.feature_extractor.sampling_rate
            )[0]

            loaded_audios.append(loaded_audio)

        return loaded_audios

    def adapt_to_conversational_chat_json(self, messages: List[dict]):
        new_messages = []
        audios = []

        for message in messages:

            new_content_items = []

            content = message["content"]

            if isinstance(content, str):
                new_content_items.append({"type": "text", "text": content})

            else:
                for content_item in message["content"]:
                    new_content_item = content_item

                    type = content_item["type"]

                    if type == "audio":
                        audio_url = content_item.get("url") or content_item.get(
                            "base64"
                        )

                        new_content_item = {"type": "audio", "audio_url": audio_url}

                        audios.append(audio_url)

                    new_content_items.append(new_content_item)

            new_message = {**message, "content": new_content_items}

            new_messages.append(new_message)

        return new_messages, audios


# universal stub used by the model loader
model_cls = AudioTextToTextModelEntity
