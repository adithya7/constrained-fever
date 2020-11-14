import re
import regex
import random
import jsonlines
import copy
from tqdm import tqdm
import spacy
import string
import unicodedata
import sys


def normalize(text: str):
    return " ".join(unicodedata.normalize('NFD', text).strip().split())


def process(title, tokenize=True):
    title = normalize(title)

    title = re.sub("-LRB-", " ( ", title)
    title = re.sub('-RRB-', " ) ", title)
    title = re.sub("-LSB-", " [ ", title)
    title = re.sub('-RSB-', " ] ", title)
    title = re.sub("-COLON-", " : ", title)

    title = re.sub(" LRB ", " ( ", title)
    title = re.sub(' RRB ', " ) ", title)
    title = re.sub(" LSB ", " [ ", title)
    title = re.sub(' RSB ', " ] ", title)
    title = re.sub(" COLON ", " : ", title)

    # Add space to before and after puncuation
    if tokenize:
        title = re.sub('([!"#$%&,-./:;<=>?@\^_`{|}~])', r' \1 ', title)

    # Remove quotes etc
    title = re.sub("``", "", title)
    title = re.sub("''", "", title)
    title = re.sub('"', "", title)

    title = re.sub("-", " ", title)
    title = re.sub("_", " ", title)

    title = re.sub(r'\s+', " ", title)

    return title.strip()


def spacy_tokenize(inp):
    _spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]

    retokenized_lines = []
    pbar = tqdm(total=1)
    i, bsz = 0, 5000
    while i >= 0:
        lines = " UNIQUE_SPLITTER ".join([line.strip() for line in inp[i:i + bsz]])
        tokens = spacy_tokenizer(lines)
        lines = " ".join(tokens).split("UNIQUE_SPLITTER")
        lines = [line.strip() for line in lines]
        retokenized_lines += lines
        i += bsz
        pbar.update(bsz / len(inp))
        if i > len(inp): i = -1
    pbar.close()
    assert len(retokenized_lines) == len(inp), print(len(retokenized_lines), len(inp))
    return retokenized_lines


def replace(pattern, replacement, sentence, claim=False):
    stop_words = '(of |a |an |the ){0,1}'
    lower_pattern = pattern.lower()

    if claim:
        match1 = regex.search(rf'(?<=The {stop_words})\b{pattern} (?!{stop_words}[A-Z][^ ]*\b)', sentence)
        match2 = regex.search(rf'(?<=The {stop_words})\b{lower_pattern} (?!{stop_words}[A-Z][^ ]*\b)', sentence)

        match3 = regex.search(rf'(?<!\b[A-Z][^ ]* {stop_words})\b{pattern} (?!{stop_words}[A-Z][^ ]*\b)', sentence)
        match4 = regex.search(rf'(?<!\b[A-Z][^ ]* {stop_words})\b{lower_pattern} (?!{stop_words}[A-Z][^ ]*\b)',
                              sentence)

        if match1 is None and match2 is None and match3 is None and match4 is None:
            return sentence, False

    sentence = regex.sub(rf'(?<=The {stop_words})\b{pattern} (?!{stop_words}[A-Z][^ ]*\b)', replacement + " ", sentence)
    sentence = regex.sub(rf'(?<=The {stop_words})\b{lower_pattern} (?!{stop_words}[A-Z][^ ]*\b)', replacement + " ",
                         sentence)

    sentence = regex.sub(rf'(?<!\b[A-Z][^ ]* {stop_words})\b{pattern} (?!{stop_words}[A-Z][^ ]*\b)', replacement + " ",
                         sentence)
    sentence = regex.sub(rf'(?<!\b[A-Z][^ ]* {stop_words})\b{lower_pattern} (?!{stop_words}[A-Z][^ ]*\b)',
                         replacement + " ", sentence)

    return sentence, True


def process_partial_match(final_evidence):
    # print(final_evidence)

    # Create relevant stop word list
    stop_words = set()
    stop_words.update(["in", "a", "the", "for", "of", "on", "to", "an", "and", "at", "by", "or", "The"])

    wiki_title = final_evidence[4]
    entity = final_evidence[0]

    partial_titles = set(wiki_title.split(' '))

    if '' in partial_titles:
        partial_titles.remove('')
    for c in string.punctuation:
        if c in partial_titles:
            partial_titles.remove(c)

    for c in string.ascii_uppercase:
        if c in partial_titles:
            partial_titles.remove(c)

    # print(partial_titles)

    evidence_sent = final_evidence[2]

    # evidence_sent = replace(re.sub(" \(.*\)", "", wiki_title), entity, evidence_sent)

    for partial_title in partial_titles:
        if partial_title in stop_words:
            continue
        evidence_sent, _ = replace(partial_title, entity, evidence_sent)

    final_evidence[2] = evidence_sent

    # print(final_evidence)
    return final_evidence[:-1]


def process_brackets(wiki_titles):
    new_titles = set()
    titles_map = dict()

    for title in wiki_titles:
        new_title = re.sub(" \(.*\)", "", title)
        new_titles.add(new_title)
        titles_map[title] = new_title

    if len(new_titles) == len(wiki_titles):
        return list(new_titles), titles_map
    return wiki_titles, {x: x for x in wiki_titles}


def exact_replace_sent(sentence, wiki_titles, ents, claim=False):
    for title, ent in zip(wiki_titles, ents):
        sentence, replaced = replace(title, ent, sentence, claim)

        # Handling middle names specially. Not applicable to claims.
        title_splits = title.split(" ")
        if len(title_splits) == 2:
            # match = re.search(r"{} ".format(title_splits[0]) + r'([A-Z][^ ]* ){0,2}' +
            #                   r"{}".format(title_splits[1]), sentence)
            sentence = re.sub(r"{} ".format(title_splits[0]) + r'([A-Z][^ ]* ){0,2}' + r"{}".format(title_splits[1]),
                              ent, sentence)

    # print(replaced)

    if replaced == False:  # and match is None:
        return sentence, False
    else:
        return sentence, True


def process_sample(samples, claims):
    entities = ["ent0", "ent1", "ent2", "ent3", "ent4"]

    new_samples = list()
    for i, sample in tqdm(enumerate(samples)):

        new_sample = dict()

        wiki_titles = set()
        for j, evidence in enumerate(sample["evidence"]):
            processed_title = process(evidence[0])
            wiki_titles.add(processed_title)
            sample["evidence"][j][0] = processed_title

        wiki_titles = list(wiki_titles)
        wiki_titles, titles_map = process_brackets(wiki_titles)

        wiki_titles.sort(key=lambda x: len(x), reverse=True)

        num_titles = len(wiki_titles)
        ents = random.sample(entities, num_titles)

        new_sample["claim"], replaced = exact_replace_sent(process(claims[i], False), wiki_titles, ents, True)

        if replaced == False:
            new_samples.append(sample)
            continue

        # new_evidences have entities replaced when exactly matched
        new_evidences = list()
        for evidence in sample["evidence"]:
            new_evidence = list()

            title = titles_map[evidence[0]]
            entity = ents[wiki_titles.index(title)]

            new_evidence.append(entity)
            new_evidence.append(evidence[1])
            # print(process(evidence[2], False), wiki_titles, ents)
            evidence_sent, _ = exact_replace_sent(process(evidence[2], False), [title], [entity])
            evidence_sent, _ = exact_replace_sent(evidence_sent, wiki_titles, ents)
            new_evidence.append(evidence_sent)

            new_evidence.append(evidence[3])
            new_evidence.append(evidence[0])

            new_evidences.append(copy.deepcopy(new_evidence))

        # # final evidences have entities replaced within the evidence even when partially matched
        final_evidences = list()
        for evidence in new_evidences:
            final_evidence = copy.deepcopy(evidence)
            final_evidence = process_partial_match(final_evidence)
            final_evidences.append(final_evidence)
        new_sample["evidence"] = final_evidences

        new_sample["old_claim"] = sample["claim"]
        new_sample["old_evidence"] = sample["evidence"]
        for key, value in sample.items():
            if key not in new_sample.keys():
                new_sample[key] = value

        new_samples.append(copy.deepcopy(new_sample))

    return new_samples


def main():
    # load data
    samples = list()

    with jsonlines.open(sys.argv[1]) as file:
        for line in file:
            if len(line["evidence"]) != 0:
                samples.append(line)
            if len(samples) == 100:
                break
    print(len(samples))

    # Evidences are tokenized but claims are not, hence, spacy tokenization of claims
    claims = list()
    for sample in samples:
        claims.append(sample["claim"])
    claims = spacy_tokenize(claims)

    new_samples = process_sample(samples, claims)

    with jsonlines.open(sys.argv[2], "w") as file:
        for sample in new_samples:
            file.write(sample)


if __name__ == "__main__":
    main()
