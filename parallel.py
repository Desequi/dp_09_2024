from scipy.spatial import distance
from concurrent.futures import ProcessPoolExecutor
import csv

def create_matrix(rows, cols, value=0.):
    return [[value for _ in range(cols)] for _ in range(rows)]

def compute_distance(args):
    vid_io, vid_ch, vector, name = args
    dist = distance.euclidean(vector[vid_io], vector[vid_ch])
    if dist < 1.1:
        with open('out_dubl.csv', 'a') as f:
            f.write(f'{name[vid_io]}_dubl_{name[vid_ch]} {dist}\n')
    return dist

filename = 'out_q.csv'
with open(filename) as file:
    lines = [line.rstrip() for line in file]

name = []
vector = create_matrix(len(lines), 576)
k = 0

for line in lines:
    tmp = line.split(' ')
    name.append(tmp[0])
    for i in range(1, len(tmp)):
        vector[k][i-1] = float(tmp[i])
    k += 1

res_vec = create_matrix(k, k, 20)

args_list = [(vid_io, vid_ch, vector, name) for vid_io in range(k-1) for vid_ch in range(vid_io+1, k)]

with ProcessPoolExecutor() as executor:
    distances = list(executor.map(compute_distance, args_list))

with open('res.csv', 'a') as f:
    f.write(str(res_vec))
