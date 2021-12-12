import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Mankind is cool")

for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov, len(token.vector))