import json
from pathlib import Path


def main(path, name):
    path = Path(path)
    with open(path / (name + '.json'), 'r') as f:
        li = json.load(f)

    with open(path / (name + '.jsonl'), 'w') as f:
        for line in li:
            f.write("{}\n".format(json.dumps(line)))

    print("done")


if __name__ == "__main__":
    main('./data', 'FriendsQA')
