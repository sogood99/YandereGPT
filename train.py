import configargparse
import matplotlib.pyplot as plt
from download_data import get_dataset
from download_gpt2 import eval
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from torch.optim import AdamW


def get_argparse():
    p = configargparse.ArgParser(default_config_files=['./.my.config'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--model_name', type=str, default="gpt2", help='model name')
    p.add('--model_dir', type=Path,
          default="./models/", help='path to model')
    p.add('--log_dir', type=Path,
          default="./log/", help='path to log')
    p.add('-v', help='verbose', action='store_true')
    p.add('--test_phrase', type=str, default="Replace me by any text you'd like. Or not",
          help="test phrase for inference")
    p.add('-d', '--debug', action='store_true')
    # this option can be set in a config file because it starts with '--'

    options = p.parse_args()

    if options.v:
        print("----------")
        print(p.format_help())
        print("----------")
        # useful for logging where different settings came from
        print(p.format_values())
    return options


class LogCallback(TrainerCallback):
    train_loss = []
    val_loss = []

    def on_evaluate(self, args, state, control, **kwargs):
        # calculate loss here
        self.val_loss.append(state.log_history[-1]["eval_loss"])
        self.train_loss.append(state.log_history[-2]["loss"])

        plt.title("YandereGPT2 Fine Tune")
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.legend()
        plt.savefig(Path("./logs/train.png"))
        plt.cla()


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
                                      evaluation_strategy="epoch",
                                      logging_strategy="epoch")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[LogCallback] if options.debug else [],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval(model, tokenizer, options.test_phrase)


if __name__ == "__main__":
    main()
