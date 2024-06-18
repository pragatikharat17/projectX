


import cv2
import numpy as np

def identify_card(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a binary threshold to get a binary image
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to find the larger diamonds
    large_diamonds = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Assuming larger diamonds have an area greater than 100
            large_diamonds.append(cnt)
    
    # Determine the bounding boxes of the large diamonds
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in large_diamonds]

    # Sort bounding boxes by their positions (top to bottom, left to right)
    bounding_boxes = sorted(bounding_boxes, key=lambda x: (x[1], x[0]))

    # Determine the positions of the larger diamonds
    positions = [(box[0] // 50, box[1] // 50) for box in bounding_boxes]

    # Map positions to rank and suit
    rank_dict = {(0, 0): 'Ace', (0, 1): '2', (0, 2): '3', (0, 3): '4', (0, 4): '5', (0, 5): '6'}
    suit_dict = {(0, 0): 'Spades', (1, 0): 'Hearts', (2, 0): 'Diamonds', (3, 0): 'Clubs'}

    rank = None
    suit = None
    for pos in positions:
        if pos in rank_dict:
            rank = rank_dict[pos]
        if pos in suit_dict:
            suit = suit_dict[pos]

    # Return the card name
    return f'{rank} of {suit}'

# File paths
file_paths = [
    '/Users/pragatik/projectX/task/tc1-1.png', '/Users/pragatik/projectX/task/tc1-2.png', '/Users/pragatik/projectX/task/tc1-3.png',
    '/Users/pragatik/projectX/task/tc2-1.png', '/Users/pragatik/projectX/task/tc2-2.png', '/Users/pragatik/projectX/task/tc2-3.png'
]

# Identify each card
for file_path in file_paths:
    card_name = identify_card(file_path)
    print(card_name)
