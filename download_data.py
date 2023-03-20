import re
import pysubs2
import os
import configargparse
import urllib.request
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def get_argparse():
    p = configargparse.ArgParser(default_config_files=['./.my.config'])
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p.add('--anime', type=str, default="School Days", help='anime name')
    p.add('--link', type=str,
          default="https://kitsunekko.net/subtitles/School%20Days/[AniYoshi]_School_Days.zip", help='link to subtitle file')
    p.add('--names', type=lambda s: [item.strip() for item in s.split(',')],
          default=["Kotonoha", "Sekai"], help='anime character names')
    p.add('--data_dir', type=Path,
          default="./data/", help='path to data')
    p.add('-v', help='verbose', action='store_true')

    options = p.parse_args()

    if options.v:
        print("----------")
        print(p.format_help())
        print("----------")
        print(p.format_values())
    return options


def get_dataset(anime_name="School Days", data_dir=Path("./data/")):
    dataset = load_dataset("text", data_files={"train": str(Path(data_dir, anime_name, "train_dataset.txt")), "test": str(
        Path(data_dir, anime_name, "test_dataset.txt"))}, sample_by="line")

    return dataset


def main():
    options = get_argparse()
    zipfile_path = os.path.join(
        options.data_dir, options.anime + ".zip")
    folder_path = os.path.join(options.data_dir, options.anime)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    urllib.request.urlretrieve(options.link, zipfile_path)

    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    ass_files = os.listdir(folder_path)

    data = []
    for file in ass_files:
        if not file.endswith(".ass"):
            continue
        subs = pysubs2.load(os.path.join(folder_path, file), encoding="utf-8")
        for line in subs:
            if line.name in options.names:
                text = re.sub(r"{.*}", "", line.text).strip()
                if text != "":
                    data.append(text)
    train, test = train_test_split(data, test_size=0.1)

    with open(os.path.join(folder_path, "train_dataset.txt"), 'w') as f:
        train_data = ""
        for line in train:
            train_data += (line + "\n")

        f.write(train_data)

    with open(os.path.join(folder_path, "test_dataset.txt"), 'w') as f:
        test_data = ""
        for line in test:
            test_data += (line + "\n")

        f.write(test_data)

    print("Finished writing train test data, train {}, test {}".format(
        len(train), len(test)))


if __name__ == "__main__":
    main()
