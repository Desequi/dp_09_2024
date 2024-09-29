# Поиск дубликатов при помощи sift. тест по train_data_yappy
# ОТкрытие видеофайла
# Получение ключевых кадров из метаданных
# Если ключевые кадры в видео расставлены равномерно по времени, а не по изменениям сцен - переразметка ключевый кадров по последовательности СКО кадров из метаданных
# Чтение из видео ключевых кадров
# Получение sift точек
# Сравнение точек текущего кадра со всеми точками из БД
# выбор дубликата по максимальному совпадению кадров
# Расширение БД уникальными видео
# Формирование выходной csv таблицы для сравнения со входной


import csv
import datetime
import os
import get_i_frame
import cv2
import sift_controller
import utils
import numpy as np

# подаем на вход train базу
#input_csv_filename = 'd:\\yappi\\train_data_yappy\\train.csv'
input_csv_filename = 'd:\\yappi\\test_data_yappy\\test.csv'

# Читаем ее полностью
with open(input_csv_filename, newline='') as f:
    reader = csv.reader(f)
    input_csv_table = list(reader)
#print(input_csv_table)

treshold = 0.25 # Порог похожести видео (сколько точек схоже на двух картинках. отнормированно к общему числу точек)
file_ext = ".mp4"
#file_dir = "d:\\yappi\\train_data_yappy\\train_dataset"
file_dir = "d:\\yappi\\test_data_yappy\\test_dataset"
output_csv_filename = 'output_csv.csv'

f = open(output_csv_filename,'w')
f.write("created,uuid,link,is_duplicate,duplicate_for,is_hard\n")
csv_bd = []
bd = []
# Читаем каждый файл из датасета
for idx, table_row in enumerate(input_csv_table):
    if idx==0:
        continue
    print(idx)
    datetime_string = table_row[0]
    datetime_obj = datetime.datetime.strptime(table_row[0], "%Y-%m-%d %H:%M:%S")

    uuid = table_row[1]
    filename = os.path.join(file_dir, uuid + file_ext)
    if not os.path.isfile(filename):
        continue

    link = table_row[2]
    #is_duplicate = bool(table_row[3])
    #duplicate_for = table_row[4]
    #is_hard = bool(table_row[5])

    # Получим номера ключевых кадров
    index = get_i_frame.extract_key_frames(filename)
    if len(index)>2:
        index = index[1:-1]

    number_center_ind = len(index) // 2
    left = number_center_ind-1
    right = number_center_ind+1
    if (left>=0) & (right<len(index)):
        index = [index[left], index[number_center_ind], index[right]]

    # # Если ключевых кадров слишком много, прорядим их
    # max_num_frames = 3
    # if len(index) > max_num_frames:
    #     step = len(index) // max_num_frames
    #     index = [index[i] for i in range(0,  len(index), step)][:max_num_frames]

    # Получим точки для каждого ключевого кадра
    sift = sift_controller.SIFT()
    max_match_array = []
    for i in index:
        cap = cv2.VideoCapture(filename)

        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if i >= 0 & i <= totalFrames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            # cv2.imshow("Video", frame)
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break
            feature = sift.extract(frame) # Получим точки

            # Если что-то пошло не так при извлечении точек
            if np.size(feature) == 1:
                if feature == None:
                    continue
            if np.size(feature) == 0:
                continue
            if np.size(feature) == 128:
                continue
            if (feature.all()) == None:
                continue

            # sift.dump_feature_frame(uuid+'_'+i  ,feature)
            if len(bd)==0: # Это кадры самого первого видео, они все уникальны, сохраним их фичи в бд
                bd.append([uuid,feature])
            else:
                match_list = sift.compare_with_db(bd, uuid, feature)
                if len(match_list) == 0:
                    bd.append([uuid, feature])
                else:
                    max_match = utils.get_top_k_result(match_list,1) # получим максимально близкий uuid родителя и метрику схожести
                    m = max_match[0]
                    max_match_array.append(m)
                    if m[1]<treshold: # кадр уникальный, сохраним его фичи в БД
                        bd.append([uuid, feature])

    if idx==1: # это первое видео, сохраним его в csv
        csv_out = [datetime_string, uuid, link, False, '', False]
        csv_bd.append(csv_out)
        str_out = (",".join(str(x) for x in csv_out))
        f.write("%s\n" % str_out)

    if len(max_match_array)>0:
        m = utils.get_top_k_result(max_match_array, 2) # выберем кадр, который больше всех совпадает с кадром в бд
        if len(m)==2:
            m1 = m[0]
            m2 = m[1]
            if (m1[0] == m2[0]) & (m1[1]>treshold) & (m2[1]>treshold):
                is_duplicate_cur = True
                is_hard_cur = False
                uidd_dublicate = m1[0]
            else:
                is_duplicate_cur = False
                is_hard_cur = False
                uidd_dublicate = ''
        else:
            m=m[0]
            if m[1] < treshold: # Кадры отличаются
                is_duplicate_cur = False
                is_hard_cur = False
                uidd_dublicate = ''
            else: # Нашли дубликат
                is_duplicate_cur = True
                is_hard_cur = False
                uidd_dublicate = m[0]
        csv_out = [datetime_string, uuid, link, is_duplicate_cur, uidd_dublicate, is_hard_cur] # сохраним в таблицу
        csv_bd.append(csv_out)

        str_out = (",".join(str(x) for x in csv_out))
        f.write("%s\n" % str_out) # Сохраним на диск новую запись
f.close()
