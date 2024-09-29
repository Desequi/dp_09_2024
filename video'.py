from scipy.spatial import distance

def create_matrix(rows, cols, value=0.):
    return [[value for _ in range(cols)] for _ in range(rows)]



filename = 'out_q4.csv'
with open(filename) as file:
    lines = [line.rstrip() for line in file]
name = []
vector = create_matrix(len(lines), 576)
k=0
for line in lines:

    tmp = line.split(' ')
    name.append(tmp[0])
    for i in range(1,len(tmp)):
        vector[k][i-1]=float(tmp[i])
    k=k+1
res_vec = create_matrix(k, k, 20)
dubl = [[]]
for vid_io in range(k-1):
    # dubl_ = []
    print(vid_io)
    for vid_ch in range(vid_io+1,k):
        if name[vid_io][0: 20] != name[vid_ch][0: 20]:
            res_vec[vid_io][vid_ch] = distance.euclidean(vector[vid_io], vector[vid_ch])
            if res_vec[vid_io][vid_ch] <0.4:
                f = open('out_dubl.csv', 'a')
                f.write(name[vid_io]+'_dubl_'+name[vid_ch]+'_'+str(res_vec[vid_io][vid_ch])+'\n')
                f.close()

f = open('res.csv', 'a')
f.write(str(res_vec))
f.close()