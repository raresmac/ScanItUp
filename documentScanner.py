# import the necessary packages
from utilities.pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils

# Parse arguments
def argumentParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = vars(ap.parse_args())
    return args
args = argumentParser()

# Load the image and compute the ratio of the old height, then resize it
def resizeImage(args):
    image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    return image, orig, ratio
image, orig, ratio = resizeImage(args)

# Convert the image to grayscale, blur it, and find edges in the image
def grayScale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return edged
edged = grayScale(image)

def showImage(image, edged):
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# showImage(image, edged)

# Find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
def contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        print("No contour detected")
        exit()
    return screenCnt
screenCnt = contour(edged)

def showOutline(image):
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# showOutline(image)

# Apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
def blackAndWhite(warped):
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    return warped
image_bw = blackAndWhite(warped)

# Show the final images
def ShowFinalImages(warped, images):
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height=650))
    for changed_image in images:
        cv2.imshow("Scanned", imutils.resize(changed_image, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
ShowFinalImages(warped, [image_bw])
