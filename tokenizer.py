import pandas as pd
from collections import Counter
import re
from fugashi import Tagger
from pykakasi import kakasi

lbase_dir = "./language-base/"
lmod_dir = "./language-models/"

def build_language_model(word_list, language_name="language"):
    total_words = len(word_list)
    counter = Counter(word_list)
    top20 = counter.most_common(20)

    model = []
    for word, count in top20:
        prob = count / total_words
        model.append([word, count, round(prob, 6)])

    df = pd.DataFrame(model, columns=["Word", "Frequency", "Probability"])
    df.to_csv(f"{lmod_dir}{language_name}_model.csv", index=False)
    print(f"{language_name} model saved!")

def load_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    text = re.sub(r'[^a-zA-Zñáéíóúü\s]', '', text)
    words = text.split()
    return words

def get_top_words(words, top_n=20):
    counter = Counter(words)
    return counter.most_common(top_n)

def count_ascii_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    text = re.sub(r'[^a-zA-Zñáéíóúü\s]', '', text)
    words = text.split()
    return len(words)

def is_japanese_line(line):
    japanese_chars = re.findall(r'[\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', line)
    return len(japanese_chars) / (len(line) + 1) > 0.5  # threshold

def is_punctuation(token):
    return re.match(r'^[\u3000-\u303F\uFF00-\uFFEF\s\・\.\,\!\?\"\'\“\”\‘\’\《\》\「\」\[\]\(\)]+$', token)


# ! Purely for handling Japanese text
with open("./language-base/japanese.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

clean_lines = [line.strip() for line in lines if is_japanese_line(line)]
cleaned_text = "\n".join(clean_lines)

tagger = Tagger()
tokens = [word.surface for word in tagger(cleaned_text)]

filtered_tokens = [token for token in tokens if token.strip() and not is_punctuation(token)]

kakasi = kakasi()
romaji_tokens = []

for token in filtered_tokens:
    results = kakasi.convert(token)
    for item in results:
        romaji = item["hepburn"].strip().lower()
        if romaji:
            romaji_tokens.append(romaji)
# ! End Japanese text handling

# Process text files
tagalog_words = load_and_tokenize('./language-base/tagalog.txt')
tagalog_top20 = get_top_words(tagalog_words)

english_words = load_and_tokenize('./language-base/english.txt')
english_top20 = get_top_words(english_words)

japanese_text = " ".join(romaji_tokens)
japanese_top20 = get_top_words(japanese_text.split())

spanish_words = load_and_tokenize('./language-base/spanish.txt')
spanish_top20 = get_top_words(spanish_words)

build_language_model(tagalog_words, "tagalog")
build_language_model(english_words, "english")
build_language_model(japanese_text.split(), "japanese")
build_language_model(spanish_words, "spanish")
