# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import glob
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="/home/intwis100/Downloads/face-alignment/shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
#intwis100@fa3a9455f625:~/Downloads/testimage
ap.add_argument("-i", "--imagepath", default="/home/intwis100/Downloads/testimage/",
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
output_dir = "./test_img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


txtfiles = []
for file in glob.glob(args['imagepath']+"/*.jpg"):
	txtfiles.append(file)
	#load the input image, resize it, and convert it to grayscale
	#image = cv2.imread(args["image"])
	image = cv2.imread(file)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	#cv2.imshow("Input", image)a
	rects = detector(gray, 2)



	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)

		import uuid
		f = str(uuid.uuid4())
		faceAligned32 = imutils.resize(faceAligned, width=32)
		ret = cv2.imwrite(output_dir+"/" + f + ".png", faceAligned32)
		print(ret)

		# display the output images
		#cv2.imshow("Original", faceOrig)
		#cv2.imshow("Aligned", faceAligned)
		#cv2.waitKey(0)