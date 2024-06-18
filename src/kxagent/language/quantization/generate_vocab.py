if __name__ == "__main__":
    import argparse
    import json

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.model_dir + "/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    vocab = tokenizer.vocab

    if config["vocab_size"] > tokenizer.vocab_size:
        META_TOKEN = "▁▁"
        for i in range(config["vocab_size"] - tokenizer.vocab_size):
            token = "{}{}".format(META_TOKEN, i)
            vocab[token] = tokenizer.vocab_size + i

    with open(args.model_dir + "/vocab.json", "w") as vocab_file:
        json.dump(vocab, vocab_file)

    print("Vocab file generated at {}".format(args.model_dir + "/vocab.json"))
