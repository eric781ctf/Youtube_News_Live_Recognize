import colorsys
import PIL.Image as Image
import cv2 as cv
import csv
import numpy as np
import pytesseract
import API_identity as API
import os

#Calculate the RGB
def Get_RGB(PIC):
	max_score = 0.0001
	dominant_color = None
	for count,(r,g,b) in PIC.getcolors(PIC.size[0]*PIC.size[1]):
		#Turn to HSV
		saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
		y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
		y = (y-16.0)/(235-16)
		#Ignore high light
		if y > 0.9:
			continue
		score = (saturation+0.1)*count
		if score > max_score:
			max_score = score
			dominant_color = (r,g,b)
	return dominant_color

#start to identify the picture RGB
def Recognize_Main(PIC,folder):
	print("Start working...")
	image = Image.open(folder + PIC)
	image_rgb = image.convert('RGB')

	#According to the different channel and different color background of titles using different method
	RGB = Get_RGB(image_rgb)
	if(RGB[0] <= 255 and RGB[0] >= 180) and (RGB[1] <= 255 and RGB[1] >= 190) and (RGB[2] <= 12 and RGB[2] >= 0):
		print("Use process_PIL method：Yellow")
		Identity_text = API.detect_text(folder + PIC)
		print(Identity_text)
		delete_img(PIC,folder)
		return Identity_text
	else:
		if folder == "CTI_Picture/":
			Identity_text = API.detect_text(folder + PIC)
			print(Identity_text)
			delete_img(PIC,folder)
			return Identity_text
		else:
			Identity_text = API.detect_text(folder + PIC)
			print(Identity_text)
			delete_img(PIC,folder)
			return Identity_text

#Used PIL module processs picture and identify the word in PIC
#Black word set as (0,1) otherwise set as (1,0)
def process_PIL(PIC,x,y,folder):
	image = Image.open(folder + PIC)
	image = image.convert('L')
	#OCR to binary
	threshold = 100
	table = []
	for i in range(256):
		if i < threshold:
			table.append(x)
		else:
			table.append(y)
	image = image.point(table, "1")
	
	image.save(folder + "process-" + PIC,"png")
	Save_image = folder + "process-" + PIC
	#Goolge API turn PIC to text
	text = API.detect_text(Save_image) 
	return text 

#Used OpenCV module process picture which background is yellow and identify the word in PIC
def process_CV_Yellow(PIC,folder):
	img = cv.imread(folder + PIC)
	#convert to gray scale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	cv.imwrite(folder + "process-" + PIC,gray)
	Save_image = folder + "process-" + PIC
	#Goolge API turn PIC to text
	text = API.detect_text(Save_image) 
	return text

#Used OpenCV module process picture and identify the word in PIC
def process_CV(PIC,folder):
	img = cv.imread(folder + PIC)
	#convert to gray scale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# threshhold
	ret, bin = cv.threshold(gray, 245, 255, cv.THRESH_BINARY)
	# closing
	kernel = np.ones((3, 3), np.uint8)
	# invert black/white
	closing = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)
	inv = cv.bitwise_not(closing)
	cv.imwrite(folder + "process-" + PIC,inv)
	Save_image = folder + "process-" + PIC
	#Goolge API turn PIC to text
	text = API.detect_text(Save_image) 
	return text

#Delete the photo to avoid it use too much space
def delete_img(PIC,folder):
	try:
		Original_File = folder + PIC
		os.remove(Original_File)
	except:
		print("Can't delete")
#Save the result text into .CSV file
def save_as_CSV(channel,Time_stamp,TextType,Identity_text):
	file = 'Result_of_'+channel+'_LIVE.csv'
	with open(file,'a', newline='',encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		try:
			writer.writerow([Time_stamp,channel,TextType,Identity_text])
		except:
			print("Something wrong")
