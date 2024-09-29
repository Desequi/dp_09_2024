import kadr


data = {}
with open('output.csv', 'r') as f:
    reader = f.readlines()
    for row in reader:
        data_spl = row.split(' ')
        name = data_spl[0].replace('.mp4', '')
        data[name] = [int(d) for d in data_spl[1:]]

for key, value in data.items():

   # kadr.read_specific_frame_cnn(f'D:/хакатоны/Цифровой прорыв24/train_data_yappy/train_dataset/{key}.mp4', value)
   kadr.save_specific_frame(f'D:/хакатоны/Цифровой прорыв24/train_data_yappy/train_dataset/{key}.mp4', value, key)
