import configargparse
import time
from download_data import get_dataset
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.optim import AdamW


def get_argparse():
    p = configargparse.ArgParser(default_config_files=['./.my.config'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--model_name', type=str, default="gpt2", help='model name')
    p.add('--model_dir', type=Path,
          default="./models/", help='path to model')
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
    model_dir = Path(options.model_dir, options.model_name)
    output_dir = Path(options.model_dir, options.model_name + "_fined_tuned")

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    dataset = get_dataset()

    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(output_dir=str(output_dir),
                                      overwrite_output_dir=True,
                                      num_train_epochs=10,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=64,
                                      eval_steps=400,
                                      save_steps=800,
                                      warmup_steps=500,
                                      evaluation_strategy="epoch")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

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
