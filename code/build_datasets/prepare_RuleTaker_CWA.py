"""
Script to build RuleTaker-CWA dataset from the original RuleTaker dataset.
Download the original RuleTaker dataset (Clark et al. 2020) from 
http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip
"""

import sys
import json
from collections import OrderedDict
import copy

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

with open(OUT_JSON, "w") as wf:
    for idx in orig_dict:
        triples = orig_dict_meta[idx]["triples"]
        rules = orig_dict_meta[idx]["rules"]
        questions_meta = orig_dict_meta[idx]["questions"]
        questions = orig_dict[idx]["questions"]
        out_dict = copy.deepcopy(orig_dict[idx])
        for q, q_out in zip(questions, out_dict["questions"]):
            meta_qidx = q["meta"]["Qid"]
            assert (
                q["text"] == questions_meta[meta_qidx]["question"]
            ), "mismatch in question text"
            if q["meta"]["strategy"] in ["rconc", "inv-rconc", "random", "inv-random"]:
                assert "CWA" in questions_meta[meta_qidx]["proofs"], "not CWA"
                q_out["label"] = "NEI"
            else:
                if q["label"] == True:
                    q_out["label"] = "SUPPORTS"
                elif q["label"] == False:
                    q_out["label"] = "REFUTES"
        wf.write(json.dumps(out_dict))
        wf.write("\n")
