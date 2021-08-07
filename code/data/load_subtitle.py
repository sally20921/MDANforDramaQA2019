import os
import json
from pathlib import Path

from tqdm import tqdm


def load_subtitle(subtitle_path):
    subtitle_path = Path(subtitle_path)
    '''
    paths = list(subtitle_path.glob('*.json'))
    if len(paths) == 0:
        paths = list(subtitle_path.glob('*/*.json'))
    '''
    subtitles = {}
    speakers = {}
    with open(str(subtitle_path), 'r') as f:
        subtitles = json.load(f)
        subtitles = {vid: ' '.join([subtitle['utter'] for subtitle in v['contained_subs']])
                     for vid, v in subtitles.items()}
    return subtitles


def update_subtitle(data_path, subtitle_path):
    suffix = '.json'
    path = Path(data_path)
    path = path.parent / (path.stem + suffix)
    if not os.path.isfile(str(path)):
        print("processing subtitle data")
        old_path = path.parent / (path.stem[:path.stem.find('_subtitle')] \
                                  + path.suffix)
        subtitles = load_subtitle(subtitle_path)
        with open(str(old_path), 'r') as f:
            data = json.load(f)
        res = []
        for row in tqdm(data):
            if row['vid'].endswith('_000'):
                # scene question
                vid = row['vid']
                vid_prefix = vid[vid.find('_000')]
                subtitle = sorted([(vid, sub) for vid, sub in subtitles.items()
                            if vid.startswith(vid_prefix)])
                subtitle = ' '.join([v[1] for v in subtitle])
                row['subtitle'] = subtitle
                '''
                speaker = sorted([(vid, sub) for vid, sub in speakers.items()
                            if vid.startswith(vid_prefix)])
                speaker = ','.join([v[1] for v in speaker])
                row['speaker'] = speaker
                '''
            else:
                # shot question
                if row['vid'] in subtitles:
                    row['subtitle'] = subtitles[row['vid']]
                else:
                    row['subtitle'] = ''
                '''
                if row['vid'] in speakers:
                    row['speaker'] = speakers[row['vid']]
                else:
                    row['speaker'] = ''
                '''
            if row['subtitle'] == '':
                row['subtitle'] = '.'  # prevent empty string
            res.append(row)

        with open(str(path), 'w') as f:
            json.dump(res, f)
