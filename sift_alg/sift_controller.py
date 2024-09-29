import cv2
import pickle
import os
import glob
from multiprocessing import Pool
import time
from utils import *
import numpy as np
import csv
import random

class SIFT():
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.threshold = 0.75
		self.indexedfolder = "d:\\yappi\\siftimg-equ"
		self.thumbfolder = "d:\\yappi\\img"

	def dump_feature_frame(self, fname, des):
		img_id = fname.split('.')[0]
		binfile = img_id + '.pkl'
		path = os.path.join(self.indexedfolder, binfile)
		with open(path, 'wb') as dumpfile:
			pickle.dump(des, dumpfile)

	def dump_eachfile(self, img_name):
		img_path = os.path.join(self.thumbfolder, img_name)
		input_img = cv2.imread(img_path, 0)
		#cv2.imshow('Original', input_img)
		input_img = cv2.resize(input_img, (480, 640))
		input_img = cv2.equalizeHist(input_img)
		mean, stddev = cv2.meanStdDev(input_img)
		if stddev>30:
			input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

		#cv2.imshow('Res', input_img)
		#cv2.waitKey(0)

		kp, des = self.sift.detectAndCompute(input_img, None)
		img_id = img_name.split('.')[0]
		binfile = img_id + '.pkl'
		path = os.path.join(self.indexedfolder, binfile) 
		with open(path, 'wb') as dumpfile:
			pickle.dump(des, dumpfile)
	
	def dump_onefile(self):
		dumpfile = open("siftdump.pkl","wb")
		for img_path in glob.glob(".\\thumb\\*.jpg"):
			img_name = img_path.split("\\")[2]
			input_img = cv2.imread(img_path, 0)
			kp, des = self.sift.detectAndCompute(input_img, None)
			img_id = img_name.split('.')[0]
			contents = {"id" : img_id, "des" : des}
			pickle.dump(contents, dumpfile)
		dumpfile.close()

	def read(self, featurepath):
		with open(featurepath, "rb") as dump:
			des = pickle.load(dump)
			if des is None:
				des = []
		return des
	
	def extract(self, input_img):
		input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
		input_img = cv2.resize(input_img, (480, 640))
		input_img = cv2.equalizeHist(input_img)
		mean, stddev = cv2.meanStdDev(input_img)
		if stddev>30:
			input_img = cv2.GaussianBlur(input_img, (3, 3), 0)

		_, des = self.sift.detectAndCompute(input_img, None)
		return des

	def compare_with_db(self, bd, uuid1, feature1):

		num_indices = 100

		if len(feature1) > num_indices:
			random_indices = random.sample(range(len(feature1)), num_indices)
			new_feature1 = [feature1[i] for i in random_indices]
			feature1 = np.array(new_feature1)

		bf = cv2.BFMatcher()
		match_list = []

		for idx2, bd2 in enumerate(bd):
			uuid2 = bd2[0]

			if uuid2 == uuid1:
				continue

			feature2 = bd2[1]
			matches = bf.knnMatch(feature1, feature2, k=2)
			similar = 0
			for m, n in matches:
				if m.distance < self.threshold * n.distance:
					similar = similar + 1
			match_list.append([uuid2, similar / len(feature1)])
		return match_list

	def search_over_all(self):
		indexed_list = os.listdir(self.indexedfolder)
		filename_csv = "output_test.csv"
		num_indices = 100

		with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
			writer = csv.writer(file, delimiter=' ')
			for idx1, feature_file1 in enumerate(indexed_list):
				feature_path1 = os.path.join(self.indexedfolder, feature_file1)
				features1 = self.read(feature_path1)
				if np.size(features1) == 0:
					continue
				if np.size(features1) == 128:
					continue
				if (features1.all()) == None:
					continue

				if len(features1) > num_indices:
					random_indices = random.sample(range(len(features1)), num_indices)
					new_features1 = [features1[i] for i in random_indices]
					features1 = np.array(new_features1)

				bf = cv2.BFMatcher()
				match_list = []
				index = feature_file1.find("_")
				feature_file1_cut = []
				if index != -1:
					feature_file1_cut = feature_file1[:index]
				start_time = time.time()
				for idx2, feature_file2 in enumerate(indexed_list):
					if idx1==idx2:
						continue
					if feature_file1_cut in feature_file2:
						continue
					# print(idx1, idx2)
					feature_path2 = os.path.join(self.indexedfolder, feature_file2)
					features2 = self.read(feature_path2)
					if np.size(features2) == 0:
						continue
					if np.size(features2) == 128:
						continue
					if (features2.all()) == None:
						continue

					if len(features2) > num_indices:
						random_indices = random.sample(range(len(features2)), num_indices)
						new_features2 = [features2[i] for i in random_indices]
						features2 = np.array(new_features2)

					matches = bf.knnMatch(features1, features2, k=2)
					similar_list = []
					for m,n in matches:
						if m.distance < self.threshold * n.distance:
							similar_list.append([m])
					match_list.append([feature_file2, len(similar_list) / len(features1)])
					del features2, similar_list
				print(time.time() - start_time)
				result = get_top_k_result(match_list=match_list, k=1)
				writer.writerow([feature_file1, result])
		#return result

	def search_over_all_fast(self):
		indexed_list = os.listdir(self.indexedfolder)
		filename_csv = "output_test.csv"
		num_indices = 100

		features = []
		feature_path = []
		for idx, feature_file in enumerate(indexed_list):
			feature_path_tmp = os.path.join(self.indexedfolder, feature_file)
			feature_path.append(feature_path_tmp)
			features.append(self.read(feature_path_tmp))


		with open(filename_csv, mode='w', newline='', encoding='utf-8') as file:
			writer = csv.writer(file, delimiter=' ')
			for idx1, features1 in enumerate(features):
				feature_file1 = feature_path[idx1]
				if np.size(features1) == 0:
					continue
				if np.size(features1) == 128:
					continue
				if (features1.all()) == None:
					continue

				if len(features1) > num_indices:
					random_indices = random.sample(range(len(features1)), num_indices)
					new_features1 = [features1[i] for i in random_indices]
					features1 = np.array(new_features1)

				bf = cv2.BFMatcher()
				match_list = []

				# FLANN_INDEX_KDTREE = 0
				# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
				# search_params = dict(checks=10)
				# flann = cv2.FlannBasedMatcher(index_params, search_params)

				index = feature_file1.find("_")
				feature_file1_cut = []
				if index != -1:
					feature_file1_cut = feature_file1[:index]
				start_time = time.time()
				for idx2, features2 in enumerate(features):
					feature_file2 = feature_path[idx2]
					if idx1==idx2:
						continue
					if feature_file1_cut in feature_file2:
						continue
					# print(idx1, idx2)
					if np.size(features2) == 0:
						continue
					if np.size(features2) == 128:
						continue
					if (features2.all()) == None:
						continue

					if len(features2) > num_indices:
						random_indices = random.sample(range(len(features2)), num_indices)
						new_features2 = [features2[i] for i in random_indices]
						features2 = np.array(new_features2)

					# matches = flann.knnMatch(features1, features2, k=2)
					matches = bf.knnMatch(features1, features2, k=2)
					# similar_list = []
					# matches = []
					similar = 0
					for m,n in matches:
						if m.distance < self.threshold * n.distance:
							similar = similar + 1
							# similar_list.append([m])
					match_list.append([feature_file2, similar / len(features1)])
					# match_list.append([feature_file2, len(similar_list) / len(features1)])
				print(time.time() - start_time)
				result = get_top_k_result(match_list=match_list, k=1)
				writer.writerow([feature_file1, result])
		#return result

	def search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		match_list = []
		if np.size(query_des) == 0:
			return match_list
		if np.size(query_des) == 128:
			return match_list
		if (query_des.all()) == None:
			return match_list
		indexed_list = os.listdir(self.indexedfolder)
		for idx, feature_file in enumerate(indexed_list):
			print(idx)
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)
			if np.size(features) == 0:
				continue
			if np.size(features) == 128:
				continue
			if (features.all()) == None:
				continue
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(query_des, features, k=2)
			similar_list = []
			for m,n in matches:
				if m.distance < self.threshold * n.distance:
					similar_list.append([m])
			match_list.append([feature_file, len(similar_list) / len(query_des)])
			del features, similar_list
		result = get_top_k_result(match_list=match_list, k=20)
		return result
	
	def measure(self, query_des, indexed_list):
		bf = cv2.BFMatcher()
		id = indexed_list[0]
		indexed_des = indexed_list[1]
		matches = bf.knnMatch(query_des, indexed_des, k=2)
		similar_list = []
		for m, n in matches:
			if m.distance < self.threshold * n.distance:
				similar_list.append([m])
		ret = [id, len(similar_list) / len(query_des)]
		del indexed_des, similar_list
		return ret
		
	def inmemory_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		pkl_file = open("siftdump.pkl", "rb")
		indexed_list = []
		for idx, contents in enumerate(pickleloader(pkl_file)):
			id, indexed_des = parse_pkl(contents)
			if (indexed_des.all()) == None:
				continue
			indexed_list.append([id, indexed_des])
		pkl_file.close()
		match_list = list(map(lambda i: self.measure(query_des, i), indexed_list))
		result = get_top_k_result(match_list=match_list, k=5)
		return result
	
	def fast_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		match_list = []
		indexed_list = os.listdir(self.indexedfolder)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=10)
		for idx, feature_file in enumerate(indexed_list):
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)
			if (features.all()) == None:
				continue
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches = flann.knnMatch(query_des, features, k=2)
			similar_list = []
			for m,n in matches:
				if m.distance < self.threshold * n.distance:
					similar_list.append([m])
			match_list.append([feature_file, len(similar_list) / len(query_des)])
			del features, similar_list
		result = get_top_k_result(match_list=match_list, k=5)
		return result
