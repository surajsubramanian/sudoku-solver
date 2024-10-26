import numpy as np
import os
import operator
import cv2
from PIL import Image

root = os.getcwd()
def imageProcessor(img_path):
    """ Here we find all possible contours inside the image by blur the image after after converting it into gray scale image. """
    image  = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, gray, contours

def bestContours(image, contours):
    """ It will return the best contour among all possible contours to crop the sudoku grid from the image. """
    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(cv2.UMat(i))
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
        c+=1
    return best_cnt

def maskCreator(gray, best_cnt, image, contours):
    """ Creates the mask which is same size as the sudoku to crop accurately."""
    mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    blur = cv2.GaussianBlur(out, (11,11), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000/2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c+=1
    return image,out

def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def boxFinder(image, out):
    """ Here we find the four corners of the sudoku inside the image."""
    final = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    box = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    gray = crop_and_warp(final, box)
    return gray

def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares

def display_rects(in_img, rects, colour=255):
    """Displays rectangles on the image."""
    img = in_img.copy()
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    return img

def main(img_path):
    """ Here we split the sudoku grid into 81 cells after performing all the below preprocessing methods. """

    image, gray, contours = imageProcessor(img_path)
    best_cnt = bestContours(image, contours)
    image,out = maskCreator(gray, best_cnt, image, contours)
    gray = boxFinder(image, out)

    squares = infer_grid(gray)
    image = display_rects(gray, squares)
    Image.fromarray(image).save('board.png')

    cut_image_grid('board.png')

def sudoku_solver(img_path):
    main(img_path)


def cut_image_grid(image_path: str, output_folder: str = "temp", grid_size: int = 9):
    # Load the image from the file
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image file not found or could not be opened.")

    # Get image dimensions
    height, width = image.shape[0], image.shape[1]
    piece_height, piece_width = height // grid_size, width // grid_size

    # Extract the base name from the image path (without extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over grid positions and save each piece
    for i in range(grid_size):
        for j in range(grid_size):
            y, x = i * piece_height, j * piece_width
            h = y + piece_height if i < grid_size - 1 else height
            w = x + piece_width if j < grid_size - 1 else width

            # Crop the piece from the original image
            piece = image[y:h, x:w]
            piece[piece > 60] = 255

            # Construct filename for each piece
            piece_filename = f"{base_name}_{i+1:02}_{j+1:02}.png"
            piece_path = os.path.join(output_folder, piece_filename)

            # Save the image piece
            cv2.imwrite(piece_path, piece)

    print(f"Image pieces saved to {output_folder}")
