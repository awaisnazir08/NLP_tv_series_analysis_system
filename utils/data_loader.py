from glob import glob
import pandas as pd


def load_subtitles_dataset(dataset_path):
    # path = (dataset_path + '/*.ass')
    subtitles_paths = glob(dataset_path)
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