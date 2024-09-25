from glob import glob
import pandas as pd


def load_subtitles_dataset(dataset_path):
    checker = dataset_path.split('.')[-1]
    if checker == 'ass':
        subtitles_paths = [dataset_path]
    else:
        subtitles_paths = glob(dataset_path + '/*.ass')
    scripts = []
    episode_numbers = []
    for path in subtitles_paths:
        # Read lines
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [','.join(line.split(',')[9:]) for line in lines]
            lines = [line.replace('\\N', ' ') for line in lines]
        
        script = " ".join(lines)    
        scripts.append(script)
        
        episode_number = int(path.split('-')[-1].split('.')[0].strip())
        episode_numbers.append(episode_number) 
    
    df = pd.DataFrame.from_dict({'episode':episode_numbers, 'script':scripts})
    
    return df