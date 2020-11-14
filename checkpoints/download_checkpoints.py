#############################################
# USAGE
# -----
# python download_checkpoints.py
#############################################

# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

import requests
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
    else:
        print(f"{path_} already exists")


if __name__ == "__main__":

    print("Downloading several models. Can cost approximately 4.6 GB disk space."
          "Please comment out checkpoints in `download_checkpoints.py` if you do not wish to download all of them.")

    """ fever_models """

    # 3 BERT-concat models
    base_path = "./fever_models/bert"

    save_path = os.path.join(base_path, "baseline")
    create_paths(save_path)
    download_file_from_google_drive("155mCp6SiYt5-De7XT1_TRG7fJOSMEK91", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('10j7BrKWX8X8rdIhFcigzKhlWSHqudVv5', os.path.join(save_path, "config.json"))

    save_path = os.path.join(base_path, "cwa")
    create_paths(save_path)
    download_file_from_google_drive("1xHx64G_TqpKfgJFPPPDhhnIvrnyA9EgO", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1m1nNtGrbp9nuAtqSGu-e8W0k7zSyEvUz', os.path.join(save_path, "config.json"))

    save_path = os.path.join(base_path, "skip-fact")
    create_paths(save_path)
    download_file_from_google_drive("1boLcwu9sYIC4H7rDkbR2XfNEUIqvzyd5", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1J_et_YS_7i3BikAdYhxGIWgriejduEPa', os.path.join(save_path, "config.json"))

    # 3 KGAT models
    base_path = "./fever_models/kgat"

    save_path = os.path.join(base_path, "baseline")
    create_paths(save_path)
    download_file_from_google_drive("1JRgWk47lDm-a4NiMYY0jRD6sZMSuq2ll", os.path.join(save_path, "model.best.pt"))

    save_path = os.path.join(base_path, "cwa")
    create_paths(save_path)
    download_file_from_google_drive("1GlDw2lriVpNdbh-QFNRghI5cklGzIUmh", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1PmbujfNQVuXKn4MlwudioYtTqnI9hair', os.path.join(save_path, "train_log.txt"))

    save_path = os.path.join(base_path, "skip-fact")
    create_paths(save_path)
    download_file_from_google_drive("1S3MlQp_PtAvfAJ3N7FoBmkJVZ-sAe_Mr", os.path.join(save_path, "model.best.pt"))
    download_file_from_google_drive('1_JCrn44wRN3XLViEgjMT5Zbtsje-314Y', os.path.join(save_path, "train_log.txt"))

    # 3 Transformer-XH models
    base_path = "./fever_models/transformer-xh"

    save_path = os.path.join(base_path, "baseline")
    create_paths(save_path)
    download_file_from_google_drive("1toNqnm8j4SdaWPMXdZ-gKWzVYZaOyHmg", os.path.join(save_path, "model_finetuned_epoch_0.pt"))

    save_path = os.path.join(base_path, "cwa")
    create_paths(save_path)
    download_file_from_google_drive("1gByjMlNhxwdcEd3bSgQU7Ewd5WiDd6uT", os.path.join(save_path, "model_finetuned_epoch_0.pt"))

    save_path = os.path.join(base_path, "skip-fact")
    create_paths(save_path)
    download_file_from_google_drive("1kQKbvD156UCZ1b0tPISL9KY3RcJoThnL", os.path.join(save_path, "model_finetuned_epoch_0.pt"))

    """ ruletaker_pretrained_models """

    base_path = "./ruletaker_pretrained_models"

    save_path = os.path.join(base_path, "ruletaker-skip-fact")
    create_paths(save_path)
    download_file_from_google_drive("1eQ_lxtQKC8-eux7frGVqGWppNxrLsaHF", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1PW9SJVwDzH2_ZuQTzTPfFSrr6ZhfcWHn', os.path.join(save_path, "config.json"))

    save_path = os.path.join(base_path, "ruletaker-cwa")
    create_paths(save_path)
    download_file_from_google_drive("1e0169sHyJl9HotyNG-q2T-lq4vfyJC_U", os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1GoTCdn66lO2UVwZNCM1X4n4aeRD51WyL', os.path.join(save_path, "config.json"))

    """ race_pretrained_models """

    save_path = "./race_pretrained_models/race_base"
    create_paths(save_path)
    download_file_from_google_drive('1pZ95ugsPZdSclRN0nH9KyUZnyrDUNiOC', os.path.join(save_path, "vocab.txt"))
    download_file_from_google_drive('1d3C8079aaw-3OWR13GRQa8oQA-Je6SZS', os.path.join(save_path, "training_args.bin"))
    download_file_from_google_drive('1XcH4pMc21ZgjkZRfbzlCyv_Zd1ieDhoe', os.path.join(save_path, "tokenizer_config.json"))
    download_file_from_google_drive('1-F0FwE74hJUa1NNBmhq8sCfD7b686MDa', os.path.join(save_path, "special_tokens_map.json"))
    download_file_from_google_drive('1HkEzcUC3gik00s8yM6LI-aUhqd7I-9pW', os.path.join(save_path, "pytorch_model.bin"))
    download_file_from_google_drive('1-gYkeLLn-zSJeepZN6xKHr7rqtUuFaW6', os.path.join(save_path, "eval_results.txt"))
    download_file_from_google_drive('1eS_X1H1KA_bldgvcitWylGy4wxZG96C4', os.path.join(save_path, "config.json"))
