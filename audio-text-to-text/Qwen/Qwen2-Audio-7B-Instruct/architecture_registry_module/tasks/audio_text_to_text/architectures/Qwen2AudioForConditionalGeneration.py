from typing import List, Optional, Union
from dataclasses import dataclass
from io import BytesIO

from urllib.request import urlopen
import librosa
import numpy as np

from architecture_registry_module.tasks.audio_text_to_text.model_entity import (
    AudioTextToTextModelEntity,
)

from transformers import (
    Qwen2AudioForConditionalGeneration as Qwen2AudioForConditionalGenerationBaseClass,
)

from transformers.feature_extraction_utils import BatchFeature

from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
)


@dataclass
class Qwen2AudioForConditionalGeneration(AudioTextToTextModelEntity):
    @classmethod
    def load_model_from_model_id(cls, model_id: str, **kwargs):
        model = Qwen2AudioForConditionalGenerationBaseClass.from_pretrained(
            model_id, **kwargs
        )
        return model

    pass

    def generate(self, text, audios, **kwargs):
        kwargs = {**kwargs, "streamer": kwargs.get("streamer")}

        audios = list(
            map(
                lambda audio_url: librosa.load(
                    BytesIO(urlopen(audio_url).read()),
                    sr=self.processor.feature_extractor.sampling_rate,
                )[0],
                audios,
            )
        )

        self.processor.__call__ = processor_monkey_patch.__get__(
            self.processor, type(self.processor)
        )

        if not audios:
            audios = None

        inputs = self.processor.__call__(
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


# TODO this is no longer needed on the latest versions of transformers. Remove it when we upgrade to ^4.49.0
def processor_monkey_patch(
    self,
    text: Union[
        TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
    ] = None,
    audios: Union[np.ndarray, List[np.ndarray]] = None,
    padding: Union[bool, str, PaddingStrategy] = False,
    sampling_rate: Optional[int] = None,
    **kwargs,
) -> BatchFeature:
    """
    Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
    and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
    the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
    WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
    of the above two methods for more information.

    Args:
        text (`str`, `List[str]`, `List[List[str]]`):
            The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
            (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
            `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        audios (`np.ndarray`, `List[np.ndarray]`):
            The audio or batch of audios to be prepared. Each audio can be a NumPy array.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                lengths).
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
    """

    if text is None:
        raise ValueError("You need to specify either a `text` input to process.")
    elif isinstance(text, str):
        text = [text]
    elif not isinstance(text, list) and not isinstance(text[0], str):
        raise ValueError(
            "Invalid input text. Please provide a string, or a list of strings"
        )

    # ensure we have as much audios as audio tokens
    num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
    num_audios = 1 if type(audios) == np.ndarray else audios is not None and len(audios)
    if num_audio_tokens != num_audios:
        raise ValueError(
            f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
        )

    if audios is not None:
        audio_inputs = self.feature_extractor(
            audios,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            **kwargs,
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop(
            "attention_mask"
        )  # rename attention_mask to prevent conflicts later on

        expanded_text = []
        audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

        for sample in text:
            replace_str = []
            while self.audio_token in sample:
                audio_length = audio_lengths.pop(0)
                input_length = (audio_length - 1) // 2 + 1
                num_audio_tokens = (input_length - 2) // 2 + 1

                expanded_audio_token = self.audio_token * num_audio_tokens

                audio_token_start_idx = sample.find(self.audio_token)
                audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                has_bos = (
                    sample[
                        audio_token_start_idx
                        - len(self.audio_bos_token) : audio_token_start_idx
                    ]
                    == self.audio_bos_token
                )
                has_eos = (
                    sample[
                        audio_token_end_idx : audio_token_end_idx
                        + len(self.audio_eos_token)
                    ]
                    == self.audio_eos_token
                )

                # Check if this audio token is surrounded by bos/eos tokens
                if not has_bos and not has_eos:
                    expanded_audio_token = (
                        self.audio_bos_token
                        + expanded_audio_token
                        + self.audio_eos_token
                    )

                replace_str.append(expanded_audio_token)
                sample = sample.replace(self.audio_token, "<placeholder>", 1)

            while "<placeholder>" in sample:
                sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_text.append(sample)
        text = expanded_text

    inputs = self.tokenizer(text, padding=padding, **kwargs)

    if audios is not None:
        inputs.update(audio_inputs)

    return BatchFeature(data={**inputs})


# universal stub used by the model loader
model_cls = Qwen2AudioForConditionalGeneration
