# Поиск дубликатов при помощи sift. тест по train_data_yappy
# ОТкрытие видеофайла
# Получение ключевых кадров из метаданных
# Если ключевые кадры в видео расставлены равномерно по времени, а не по изменениям сцен - переразметка ключевый кадров по последовательности СКО кадров из метаданных
# Чтение из видео ключевых кадров
# Получение sift точек
# Сравнение точек текущего кадра со всеми точками из БД
# выбор дубликата по максимальному совпадению нескольких кадров
# Расширение БД уникальными видео

import datetime
import os
import get_i_frame
import cv2
import sift_controller
import utils
import numpy as np
import pickle
import urllib.request

def find_dublicate_sift(url):
    file_dir = "d:\\yappi\\test_data_yappy\\test_dataset"
    base_fname = "d:\\yappi\\test_data_yappy\\test_dataset\\siftdump.pkl"

    filename = url.split('/')[-1]
    uuid = url.split('.')[0]
    filename = os.path.join(file_dir, filename)

    urllib.request.urlretrieve(url, filename)

    if not os.path.isfile(filename):
        print("error open file!")
        return [False, '']

    treshold = 0.15 # Порог похожести видео (сколько точек схоже на двух картинках. отнормированно к общему числу точек)

    # подаем на вход train базу
    with open(base_fname, "rb") as dump:
        bd = pickle.load(dump)
        if bd is None:
            bd = []

    bd_len_orig = len(bd)

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

    is_duplicate_cur = False
    is_hard_cur = False
    uidd_dublicate = ''

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

    if (bd_len_orig < len(bd)):
        with open(base_fname, 'wb') as dumpfile:
            pickle.dump(bd, dumpfile)

    return [is_duplicate_cur, uidd_dublicate]



if __name__=="__main__":
    url = 'https://s3.ritm.media/yappy-db-duplicates/5150146c-2ebe-48f4-84ef-363d6fb73b5b.mp4'
    isdublicate, uuid = find_dublicate_sift(url)
    print(isdublicate, uuid)