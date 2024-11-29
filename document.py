import cv2
from PIL import Image
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def crop_document(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded")
    
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    # Find the contours in the edged image, keeping only the largest ones
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Loop over the contours to find the document
    screenCnt = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        raise ValueError("No document found")

    # Apply the perspective transform to obtain a top-down view of the document
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    return warped

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image.astype(np.uint8)

def invert_colors(image):
    return cv2.bitwise_not(image)

def enhance_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], 50)  # Increase saturation by 50
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image

def blue_tint(image):
    blue_filter = np.zeros_like(image)
    blue_filter[:, :, 0] = 50  # Add blue channel intensity
    blue_tinted_image = cv2.add(image, blue_filter)
    return np.clip(blue_tinted_image, 0, 255).astype(np.uint8)

def save_as_pdf(image_paths, pdf_path, color_transform=None):
    images = []
    for image_path in image_paths:
        cropped_image = crop_document(image_path)
        if color_transform:
            cropped_image = color_transform(cropped_image)
        pil_image = Image.fromarray(cropped_image).convert('RGB')
        images.append(pil_image)

    if not images:
        raise ValueError("No images to save as PDF")

    images[0].save(pdf_path, save_all=True, append_images=images[1:])

# Paths to your images
image_paths = [r'E:\document\1.jpg', r'E:\document\2.jpg']
pdf_path = r'E:\document\scanned_document.pdf'

# Save the cropped images with a chosen color transformation as a PDF
# Examples:
# save_as_pdf(image_paths, pdf_path, convert_to_grayscale)
# save_as_pdf(image_paths, pdf_path, convert_to_sepia)
# save_as_pdf(image_paths, pdf_path, invert_colors)
# save_as_pdf(image_paths, pdf_path, enhance_colors)
save_as_pdf(image_paths, pdf_path, enhance_colors)