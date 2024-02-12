import spacy
from typing import Union
import subprocess


LEMMATIZER_SOURCE = {
    "uk": "uk_core_news_sm",
    "ru": "ru_core_news_sm",
    "en": "en_core_web_sm",
}


class Lemmatizer:
    def __init__(self, lang, verbose=False):
        """
        :param lang: should be one of ["en", "uk", "ru"]
        Note: {lang}_core_news_sm should be installed beforehand via
        python -m spacy download {source}, see LEMMATIZER_SOURCE
        """
        if lang not in LEMMATIZER_SOURCE.keys():
            raise Exception("wrong lang param")
        command = f"python -m spacy download {LEMMATIZER_SOURCE[lang]}"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if verbose:
            print(f"Command output: {result.stdout}")
        self.nlp = spacy.load(LEMMATIZER_SOURCE[lang])
    
    def get_word_bases(self, text: str) -> list[dict]:
        """
        :param text: string or list whose words should be lemmatized
        :returns ans: for each word returns a dictionary with the information about a location
            and the base form of a word
        """
        doc = self.nlp(text)
        ans = []
        for token in doc:
            base = token.lemma_
            word = token.text
            offset = token.idx
            if base.isalpha():
                ans.append({
                    "base": base,
                    "word": word,
                    "offset": offset
                })
        return ans
    
    def lemmatize_text(self, text: str) -> str:
        """
        :param text: string or list whose words should be lemmatized
        :returns bases: the same text, where every word is converted to its base
        """
        ans = []
        for word in self.get_word_bases(text):
            ans.append(word["base"])
        return " ".join(ans)