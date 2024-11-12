import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image

def preprocess_image(file_name):
    def approximate_contour(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    def get_receipt_contour(contours):    
        # Loop over the contours
        for c in contours:
            approx = approximate_contour(c)
            # If our approximated contour has four points, we can assume it's the receipt's rectangle
            if len(approx) == 4:
                return approx
        return None

    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # Top-left point has the smallest sum
        # Bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # Compute the difference between the points:
        # The top-right will have the minimum difference 
        # The bottom-left will have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect / resize_ratio

    def wrap_perspective(img, rect):
        # Unpack rectangle points: top left, top right, bottom right, bottom left
        (tl, tr, br, bl) = rect
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # Take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        # Destination points which will be used to map the screen to a "scanned" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        # Warp the perspective to grab the screen
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def bw_scanner(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 21, offset=5, method="gaussian")
        return (gray > T).astype("uint8") * 255

    # Step 1: Load and downscale the image
    img = Image.open(file_name)
    img.thumbnail((800, 800), Image.LANCZOS)
    img.show()

    image = cv2.imread(file_name)
    # Downscale image as finding receipt contour is more efficient on a small image
    resize_ratio = 500 / image.shape[0]
    original = image.copy()

    width = int(image.shape[1] * resize_ratio)
    height = int(image.shape[0] * resize_ratio)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Step 2: Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 3: Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 4: Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    # Step 5: Edge detection
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    # Step 6: Detect all contours in Canny-edged image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Step 7: Get 10 largest contours
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]



    # Step 8: Approximate the contour by a more primitive polygon shape
    receipt_contour = get_receipt_contour(largest_contours)
    if receipt_contour is None:
        print("No receipt contour detected.")
        return
    
    

    # Step 9: Perform perspective transformation
    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))
    # Step 10: Convert to black and white
    # result = bw_scanner(scanned)


    # Step 11: Save the result
    output = Image.fromarray(scanned)
    return output
