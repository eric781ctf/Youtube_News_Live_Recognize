import pafy
import time, threading
import cv2 as cv
import csv
import Method
import Classifier

#setting url and folder for each channel
url_CTI = "https://www.youtube.com/watch?v=wUPPkSANpyo&ab_channel=%E4%B8%AD%E5%A4%A9%E9%9B%BB%E8%A6%96" #中天新聞直播
url_SET = "https://www.youtube.com/watch?v=4ZVUmEUFwaY&ab_channel=%E4%B8%89%E7%AB%8BLIVE%E6%96%B0%E8%81%9E" #三立新聞直播
url_EBC = "https://www.youtube.com/watch?v=63RmMXCd_bQ&ab_channel=%E6%9D%B1%E6%A3%AE%E6%96%B0%E8%81%9ECH51" #東森新聞直播
url_TVBS = "https://www.youtube.com/watch?v=A4FbB8UhNRs&ab_channel=TVBSNEWS" #TVBS新聞直播
url_FTVN = "https://www.youtube.com/watch?v=XxJKnDLYZz4&feature=emb_title&ab_channel=%E6%B0%91%E8%A6%96%E7%9B%B4%E6%92%ADFTVNLive53" #民視新聞直播
url = [url_CTI, url_EBC, url_SET, url_TVBS, url_FTVN]

#Setting the folder where photo will be save and the stamp of each channel
folder = ["CTI_Picture/","EBC_Picture/","SET_Picture/","TVBS_Picture/","FTVN_Picture/"]
channel_name_stamp = ['CTI', 'EBC', 'SET', 'TVBS', 'FTVN']

timeout = 7200 * 60   # [seconds]
timeout_start = time.time()

threads = []

def CapturePIC(num):
	while time.time() < timeout_start + timeout:
		try:
			picture_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) #Used to named the picture
			Time_stamp = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time())) #Used to record the time into csv
			Control_Time = time.strftime("%H", time.localtime(time.time()))

			C_Name = channel_name_stamp[num]
			Pafy = pafy.new(url[num])
			best = Pafy.getbest(preftype="mp4")
			capture = cv.VideoCapture()
			capture.open(best.url)
			ret, img = capture.read()
			if Control_Time >= '11' and Control_Time <= '20':
				print("Capture PIC from :" + folder[num])
				cv.imwrite(folder[num] + C_Name + picture_time + '.png',img)
				Big_PIC = C_Name + picture_time + '.png'

				#According to different channel, there are corresponding range
				if C_Name == 'CTI':
					cropped = img[870:965, 455:1840]
				elif C_Name == 'EBC':
					cropped = img[850:970, 100:1900]
				elif C_Name == 'SET':
					cropped = img[845:955, 370:1840]
				elif C_Name == 'TVBS':
					cropped = img[860:965, 480:1840]
				elif C_Name == 'FTVN':
					cropped = img[870:970, 440:1780]

				cv.imwrite(folder[num] + 'Z' + picture_time + '.png', cropped)
				PIC =  'Z' + picture_time + '.png'

				#Check the PIC is News Title or Subtitle
				TextType = Classifier.init(PIC,folder[num])
				if TextType == 'title':
					print("Title")
					#Turn the PIC to String and save as .csv
					Method.save_as_CSV(C_Name,Time_stamp,TextType,Method.Recognize_Main(PIC,folder[num]))
					Method.delete_img(PIC,folder[num])
				else:
					print("Word")
					Method.delete_img(PIC,folder[num])
					Method.delete_img(Big_PIC,folder[num])
					print('Delete PIC')
					print(Rest_Time)
			else:
				print("It's not working time")
		except KeyboardInterrupt:
			print("---------------Stop---------------")
			sys.exit(0)
		except:
			continue
		print('------------' + C_Name + 'wait------------')
		time.sleep(5)

try:
	for x in range(5):
		threads.append(threading.Thread(target = CapturePIC, args = (x,)))
		threads[x].start()
except KeyboardInterrupt:
	print("---------------Stop---------------")
	sys.exit(0)