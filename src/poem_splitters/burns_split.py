import os
import random

def split_poems(input_path, base_output_dir, author_prefix, train_frac=0.8, seed=42):
    # Splits large Robert Burns .txt file into individual poem .txt files.
    # Randomly splits into training and testing folders based on train_frac.
    random.seed(seed)

    # read full text
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # split by lines
    lines = full_text.splitlines()

    poems = []
    current_poem = []
    blank_lines = 0

    for line in lines:
        if line.strip() == "":
            blank_lines += 1
        else:
            if blank_lines >= 2 and not line.startswith(" "):
                if current_poem:
                    poems.append("\n".join(current_poem).strip())
                    current_poem = []
            current_poem.append(line)
            blank_lines = 0

    # add last poem
    if current_poem:
        poems.append("\n".join(current_poem).strip())

    poems = [p for p in poems if len(p.strip()) > 0]

    # Shuffle poems
    random.shuffle(poems)

    # determine split index
    split_idx = int(len(poems) * train_frac)
    train_poems = poems[:split_idx]
    test_poems = poems[split_idx:]
    print(f"Total poems: {len(poems)} — Train: {len(train_poems)}, Test: {len(test_poems)}")

    # helper function to save poems
    def save_poems(poem_list, split_folder):
        out_dir = os.path.join(base_output_dir, split_folder, author_prefix.capitalize())
        os.makedirs(out_dir, exist_ok=True)

        for i, poem in enumerate(poem_list):
            out_path = os.path.join(out_dir, f"{author_prefix}_{i}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(poem)

    save_poems(train_poems, 'training')
    save_poems(test_poems, 'testing')

    print(f"Saved {len(train_poems)} training poems and {len(test_poems)} testing poems for {author_prefix.capitalize()}.")

split_poems(
    input_path="data/raw_poems/robertburns.txt",
    base_output_dir="data/raw_text",
    author_prefix="robertburns"
)
