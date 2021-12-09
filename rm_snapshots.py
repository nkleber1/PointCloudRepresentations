import os

for d in os.listdir('logging/snapshot/'):
    curr_dir = 'logging/snapshot/{}/models/'.format(d)
    files = os.listdir(curr_dir)
    for i in range(len(files)-2):
        f = files[i]
        os.remove(os.path.join(curr_dir, f))