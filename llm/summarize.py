from pathlib import Path

from transformers import pipeline, AutoTokenizer

class Summarizer:
    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        self.summarizer = pipeline("summarization", model=model_name)
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_prefix_space = True
        
        # Block common link/CTA fragments
        bad_phrases = [
            "http://", "https://", "http", "https", "://",
            "www", "www.",  # include both, punctuation may be split
            "Click here", "click here", "Click Here",
            " Click here", " click here", " Click Here",
            "Click\u00a0here", " click\u00a0here",
        ]
        
        self.bad_words_ids = self.tokenizer(
            bad_phrases,
            add_special_tokens=False
        ).input_ids
        
        print("Summerization engine loadded")


    def __del__(self):
        del self.summarizer
        
        
    def summarize(self, text, in_max_length, in_min_length = 30, in_do_sample = False):
        return self.summarizer(text, max_length=in_max_length,
                               min_length=in_min_length,
                               do_sample=in_do_sample,
                               bad_words_ids=self.bad_words_ids,
                               no_repeat_ngram_size=3,
                               early_stopping=True)