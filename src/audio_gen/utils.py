from typing import List
import os
import re
from itertools import product
from numpy import ndarray
import scipy
from dotenv import load_dotenv, find_dotenv


def get_env_vars():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path)


def generate_string_bracket_combinations(s: str) -> List[str]:
    """
    Generate all combinations of strings given the options in the brackets
    # Example usage
    >> s = "ab[c,d]ef"
    >> lst_s = generate_combinations(s)
    >> print(lst_s)
    ['abcef', 'abdef']
    #
    :param s: string with options in brackets.
    :return: list of options.
    """
    parts = re.split(r'(\[[^]]*])', s)
    for i, part in enumerate(parts):
        if part.startswith('[') and part.endswith(']'):
            options = re.sub(r',([^ ])', r', \1', part[1:-1]).split(', ')
            parts[i] = options
    combinations = product(*[p if isinstance(p, list) else [p] for p in parts])
    result = [''.join(combination).rstrip(' ') for combination in combinations]

    return result


def save_wav_audio(audio: ndarray, directory: str, name: str, rate: int):
    scipy.io.wavfile.write(filename=os.path.join(directory, name+".wav"),
                           rate=rate, data=audio)
