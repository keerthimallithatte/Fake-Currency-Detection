from utils import *
import cv2
from matplotlib import pyplot as plt
import subprocess
image = cv2.VideoCapture(0)

while(True):
	xy,test_img=image.read()
	max_val = 8
	max_pt = -1
	max_kp = 0

	orb = cv2.ORB_create()						#Oriented FAST and Rotated BRIEF
	original = resize_img(test_img, 0.4)
	(kp1, des1) = orb.detectAndCompute(test_img, None)

	training_set = ['files/test_10_1.jpg', 'files/test_10_2.jpg', 'files/test_10_3.jpg', 'files/test_10_4.jpg','files/test_10_5.jpg', 'files/test_10_6.jpg', 'files/test_20_1.jpg', 'files/test_20_2.jpg','files/test_20_3.jpg', 'files/test_20_4.jpg','files/test_20_5.jpg', 'files/test_20_6.jpg','files/test_new_500_1.jpg','files/test_new_500_2.jpg','files/test_new_500_3.jpg','files/test_new_500_4.jpg','files/test_new_500_5.jpg','files/test_new_500_6.jpg','files/test_new_2000_1.jpg','files/test_new_2000_2.jpg','files/test_new_2000_3.jpg','files/test_new_2000_4.jpg','files/test_new_2000_5.jpg','files/test_new_2000_6.jpg','files/test_old_100_1.jpg','files/test_old_100_2.jpg','files/test_old_100_3.jpg','files/test_old_100_4.jpg','files/test_old_100_5.jpg','files/test_old_100_6.jpg','files/test_old_10_1.jpg','files/test_old_10_2.jpg','files/test_old_10_3.jpg','files/test_old_10_4.jpg','files/test_old_10_5.jpg','files/test_old_10_6.jpg','files/test_old_50_1.jpg','files/test_old_50_2.jpg','files/test_old_50_3.jpg','files/test_old_50_4.jpg','files/test_old_50_5.jpg','files/test_old_50_6.jpg']

	for i in range(0, len(training_set)):				# train image
		train_img = cv2.imread(training_set[i])

		(kp2, des2) = orb.detectAndCompute(train_img, None)

		bf = cv2.BFMatcher()						# brute force matcher
		all_matches = bf.knnMatch(des1, des2, k=2)

		good = []
		# give an arbitrary number -> 0.789
		# if good -> append to list of good matches
		for (m, n) in all_matches:
			if m.distance < 0.789 * n.distance:
				good.append([m])

		if len(good) > max_val:
			max_val = len(good)
			max_pt = i
			max_kp = kp2

	if max_val != 8:
		#print(training_set[max_pt])
		print('good matches ', max_val)

		train_img = cv2.imread(training_set[max_pt])
		img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
		note = str(training_set[max_pt])[6:-4]
		print('\nDetected denomination: Rs. ', note)
		cv2.imshow("image",img3)
		#(plt.imshow('figure1',img3), plt.show())
	else:
		print('No Matches')
image.release()
cv2.destroyAllwindows()
