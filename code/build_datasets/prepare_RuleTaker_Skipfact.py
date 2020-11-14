"""
Script to build RuleTaker-Skip-fact dataset from the original RuleTaker dataset.
Download the original RuleTaker dataset (Clark et al. 2020) from 
http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip
"""

import sys, re
import json
from collections import OrderedDict
import copy
import random

# rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/train.jsonl
INP_JSON = sys.argv[1]
# rule-reasoning-dataset-V2020.2.4/depth-3ext-NatLang/meta-train.jsonl
INP_JSON_META = sys.argv[2]
OUT_JSON = sys.argv[3]

orig_dict = OrderedDict()
with open(INP_JSON, "r") as rf:
    for line in rf:
        json_dict = json.loads(line.strip())
        orig_dict[json_dict["id"]] = json_dict

orig_dict_meta = {}
with open(INP_JSON_META, "r") as rf:
    for line in rf:
        json_dict = json.loads(line.strip())
        orig_dict_meta[json_dict["id"]] = json_dict

"""
only considers `proof` and `inv-proof` strategy examples
for each example (originally true or false), identify the base triple(s) from the proof
    and create a new instance {(orig_context - triple(s)), (question)}
"""
none_triples = 0
total_count = 0
support, refute, nei = 0, 0, 0
with open(OUT_JSON, "w") as wf:
    for idx in orig_dict:
        triples = orig_dict_meta[idx]["triples"]
        rules = orig_dict_meta[idx]["rules"]
        sents = []
        sents.extend([triples[inst]["text"] for inst in triples])
        sents.extend([rules[inst]["text"] for inst in rules])
        questions_meta = orig_dict_meta[idx]["questions"]
        questions = orig_dict[idx]["questions"]
        out_dict = copy.deepcopy(orig_dict[idx])
        valid_questions = []
        alt_counter = 0
        potential_alt_pairs = []
        for q, q_out in zip(questions, out_dict["questions"]):
            meta_qidx = q["meta"]["Qid"]
            assert (
                q["text"] == questions_meta[meta_qidx]["question"]
            ), "mismatch in question text"
            if q["meta"]["strategy"] not in ["proof", "inv-proof"]:
                continue
            if "OR" in questions_meta[meta_qidx]["proofs"]:
                """ currently ignoring if we have multiple proofs for the same question """
                continue
            proof_triples = list(
                set(re.findall(r"triple[0-9]+", questions_meta[meta_qidx]["proofs"]))
            )
            if len(proof_triples) == 0:
                none_triples += 1
            total_count += 1
            for proof_triple in proof_triples:
                alt_context = ""
                alt_sentScramble = []
                for sent_idx in orig_dict[idx]["meta"]["sentenceScramble"]:
                    if sent_idx != int(proof_triple.lstrip("triple")):
                        alt_context += "%s " % sents[sent_idx - 1]
                        alt_sentScramble.append(sent_idx)
                alt_context = alt_context.rstrip()
                q_copy = copy.deepcopy(q)
                q_copy["label"] = "NEI"
                out_dict_alt = copy.deepcopy(orig_dict[idx])
                # create a new context, question pair
                alt_counter += 1
                alt_idx = "%s-ALT%d" % (idx, alt_counter)
                out_dict_alt["id"] = alt_idx
                out_dict_alt["context"] = alt_context
                out_dict_alt["meta"]["sentenceScramble"] = alt_sentScramble
                out_dict_alt["questions"] = [
                    {
                        "id": alt_idx + "-1",
                        "text": q["text"],
                        "label": "NEI",
                        "meta": q["meta"],
                    }
                ]
                # we will sample only a few of these
                potential_alt_pairs.append(json.dumps(out_dict_alt))

            # altering labels for the original questions to FEVER style
            if q["label"] == True:
                q_out["label"] = "SUPPORTS"
            elif q["label"] == False:
                q_out["label"] = "REFUTES"
            valid_questions.append(q_out)

        # we sub-sample NEI to maintain class balance
        if len(valid_questions) > 0 and len(potential_alt_pairs) > 0:
            random.shuffle(potential_alt_pairs)
            sampled_alt_pairs = potential_alt_pairs[: int(len(valid_questions) / 2)]
            for alt_pair in sampled_alt_pairs:
                wf.write(alt_pair)
                wf.write("\n")
        out_dict["questions"] = valid_questions
        wf.write(json.dumps(out_dict))
        wf.write("\n")

print(total_count)
print(none_triples)
