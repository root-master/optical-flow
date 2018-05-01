import gym
import random
import time
import cv2
import numpy as np

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (3,3),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

env = gym.make('MontezumaRevenge-v0')
old_frame = env.reset()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
error_counter = 0

for i in range(100000):
	try:
		env.render()
		frame, reward, terminal, step_info = env.step(env.action_space.sample())
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		# Select good points
		good_new = p1[st==1]
		good_old = p0[st==1]
		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new,good_old)):
		    a,b = new.ravel()
		    c,d = old.ravel()
		    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(frame,mask)
		cv2.imshow('frame',img)
		#time.sleep(1)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			cv2.destroyAllWindows()
			break
		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

		if terminal:
			print('GAME OVER!')
			env.close()
			cv2.destroyAllWindows()
			break
	except:
		print('something is wrong')
		error_counter += 1
		if error_counter > 10:
			env.close()
			cv2.destroyAllWindows()
			break



