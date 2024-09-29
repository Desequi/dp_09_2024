import pandas as pd
from imagededup.methods import CNN
from kadr import get_key_frame,read_specific_frame

data = {}
with open('output.csv', 'r') as f:
    reader = f.readlines()
    for row in reader:
        data_spl = row.split(' ')
        name = data_spl[0].replace('.mp4', '')
        data[name] = [int(d) for d in data_spl[1:]]

for key, value in data.items():

   images = read_specific_frame(f'train_data_yappy/train_dataset/{key}.mp4', value)



if __name__ == '__main__':

    cnn = CNN()

    image_ = 'mixed_images/ukbench00120.jpg'

    det = ' '
    encoded_image = cnn.encode_image(image_)
    # print(encoded_image[0][0])
    my_formatted_list = [ '%.5f' % encoded_image[0][elem] for elem in range(0, len(encoded_image[0]), 4) ]
    # print(my_formatted_list)

    str = image_+ det
    for elem in my_formatted_list:
        str = str + elem + det

    f = open('out.csv', 'a')
    f.write(str + '\n')
    f.close()



    print(str)

    # df = pd.DataFrame(encoded_image)
    # df.to_csv('encoded_images.csv', index=False)


