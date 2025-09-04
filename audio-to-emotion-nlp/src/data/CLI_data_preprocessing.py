"""Translate Polish sentences to English using a pretrained translation model."""

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def translate_texts(texts: list, model_name="Helsinki-NLP/opus-mt-pl-en") -> list:
    """Translate Polish sentences into English using a pretrained translation model.

    Args:
        texts (list of str): List of sentences in Polish to translate.
        model_name (str, optional): Name of the HuggingFace translation model.
            Defaults to "Helsinki-NLP/opus-mt-pl-en".

    Returns:
        list of str: Translated sentences in English.
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline(
        "translation", model=model, tokenizer=tokenizer, device=device
    )
    return [translator(text)[0]["translation_text"] for text in tqdm(texts)]
