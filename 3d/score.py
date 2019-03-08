import subprocess
import os

def score(molfile1, molfile2, path_to_lsalign='/data/rsg/chemistry/yangk/LSalign/src'):
    """
    LSalign similarity score for two molecules, each in a separate molfile whose path is given as input.
    """
    with open('tmp.txt', 'w') as f:
        subprocess.call([os.path.join(path_to_lsalign, 'LSalign'), molfile1, molfile2, '-rf 1'], stdout=f)
    with open('tmp.txt', 'r') as f:
        lines = f.readlines()
    score_row = lines[3].strip().split()
    pc_score_avg = (float(score_row[2]) + float(score_row[3])) * 0.5
    return pc_score_avg

if __name__ == '__main__':
    print(score('0.mol2', '1.mol2'))