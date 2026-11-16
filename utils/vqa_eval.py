"""VQA answer normalisation (v1) and accuracy.

Normalisation follows the core rules of the official VQA evaluation script: lower-case, strip, punctuation
handling (keep decimal points and digit commas), number words to digits, drop articles a/an/the, and fix common
contractions. Version "vqa-norm-v1"; the contraction list is abbreviated. The same function is applied to the
cheap answer, the full answer and the human answers, so the change indicator is consistent.

Accuracy = min(1, matches / 3) over the ten human answers (the usual approximation of the official
leave-one-out average).
"""
import re

VERSION = "vqa-norm-v1"
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "lets": "let's",
    "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "thats": "that's", "thered": "there'd", "theres": "there's", "theyd": "they'd", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _punct(text):
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or re.search(_COMMA_STRIP, text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _digit_article(text):
    words = []
    for w in text.lower().split():
        w = _NUMBERS.get(w, w)
        if w not in _ARTICLES:
            words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def normalize(ans):
    if ans is None:
        return None
    a = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    return _digit_article(_punct(a)).strip()


def accuracy(pred, human_answers):
    p = normalize(pred)
    matches = sum(normalize(h) == p for h in human_answers)
    return min(1.0, matches / 3.0)
