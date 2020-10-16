import os
#Use the private key you download from google put the .json in the same folder
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="avid-influence-291005-47bb27a19dde.json"

#Detected texts in the photo
def detect_text(PIC):
	from google.cloud import vision
	import io
	client = vision.ImageAnnotatorClient()

	with io.open(PIC, 'rb') as image_file:
		content = image_file.read()

	image = vision.types.Image(content=content)

	response = client.text_detection(image=image)

	texts = response.text_annotations
	#Make sure whether the photo we update are no words in it or not
	if not texts:
		del response
		texts = "Nothing"
		return texts
	else:
		del response
	return texts[0].description