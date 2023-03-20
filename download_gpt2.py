import os
import configargparse
import pathlib
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed


def get_argparse():
    p = configargparse.ArgParser(default_config_files=['./.my.config'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--model_name', type=str, default="gpt2", help='model name')
    p.add('--model_dir', type=pathlib.Path,
          default="./models/gpt2/", help='path to model')
    p.add('-v', help='verbose', action='store_true')
    p.add('--test_phrase', type=str, default="Replace me by any text you'd like. Or not",
          help="test phrase for inference")
    # this option can be set in a config file because it starts with '--'

    options = p.parse_args()

    if options.v:
        print("----------")
        print(p.format_help())
        print("----------")
        # useful for logging where different settings came from
        print(p.format_values())
    return options


def main():
    set_seed(42)

    options = get_argparse()

    if not os.path.exists(options.model_dir) or not os.listdir(options.model_dir):
        # empty
        print("Directory is empty")
        tokenizer = GPT2Tokenizer.from_pretrained(options.model_name)
        model = GPT2LMHeadModel.from_pretrained(options.model_name)

        if type(model) is not GPT2LMHeadModel:
            return

        os.makedirs(options.model_dir, exist_ok=True)
        tokenizer.save_pretrained(options.model_dir)
        model.save_pretrained(options.model_dir)
    else:
        print("Directory is not empty, testing")

    tokenizer = GPT2Tokenizer.from_pretrained(options.model_dir)
    model = GPT2LMHeadModel.from_pretrained(options.model_dir)

    print("Model Params: {}".format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))

    if type(model) is not GPT2LMHeadModel:
        return

    # test it out
    start_time = time.time()

    text = options.test_phrase
    print("\nInference: {}\n".format(text) + 100*'-')
    encoded_input = tokenizer(text, return_tensors='pt')
    sample_outputs = model.generate(
        **encoded_input,
        do_sample=True,
        max_length=50,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id
    )

    time_used = time.time() - start_time

    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(
            sample_output, skip_special_tokens=True)))

    print("Time Used: {:.3f}s".format(time_used))
    print(tokenizer.eos_token_id)


if __name__ == "__main__":
    main()
