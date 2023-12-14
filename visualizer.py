
import numpy as np
import cv2

_color_map_errors_kitti = np.array([
        [ 0,       0.1875, 149,  54,  49],
        [ 0.1875,  0.375,  180, 117,  69],
        [ 0.375,   0.75,   209, 173, 116],
        [ 0.75,    1.5,    233, 217, 171],
        [ 1.5,     3,      248, 243, 224],
        [ 3,       6,      144, 224, 254],
        [ 6,      12,       97, 174, 253],
        [12,      24,       67, 109, 244],
        [24,      48,       39,  48, 215],
        [48,  np.inf,       38,   0, 165]
]).astype(float)

def color_error_image_kitti(errors, scale=1, mask=None, BGR=True, dilation=1):
    errors_flat = errors.flatten()
    colored_errors_flat = np.zeros((errors_flat.shape[0], 3))
    
    for col in _color_map_errors_kitti:
        col_mask = np.logical_and(errors_flat>=col[0]/scale, errors_flat<=col[1]/scale)
        colored_errors_flat[col_mask] = col[2:]
        
    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 0

    colored_errors = colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.uint8)

    if not BGR:
        #colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]
        colored_errors = cv2.cvtColor(colored_errors, cv2.COLOR_RGB2BGR)

    if dilation>0:
        kernel = np.ones((dilation, dilation))
        colored_errors = cv2.dilate(colored_errors, kernel)
    return colored_errors


def add_text_and_rectangle(img, text, rectangle_width = 300, text_scale = 1.0):
    # Get image dimensions
    height, width, _ = img.shape

    # Define rectangle parameters
    rectangle_color = (255, 255, 255)  # White color in BGR format
    rectangle_height = 50

    # Calculate rectangle position (bottom center)
    rectangle_start_point = ((width - rectangle_width) // 2, height - rectangle_height)
    rectangle_end_point = (rectangle_start_point[0] + rectangle_width, rectangle_start_point[1] + rectangle_height)

    # Draw the rectangle on the image
    img = cv2.rectangle(img, rectangle_start_point, rectangle_end_point, rectangle_color, cv2.FILLED)

    # Define text parameters
    text_color = (0, 0, 0)  # Black color in BGR format
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 2

    # Calculate text position (centered within the rectangle)
    text_size = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
    text_position = ((width - text_size[0]) // 2, height - 20)

    # Add the text to the image
    img = cv2.putText(img, text, text_position, text_font, text_scale, text_color, text_thickness)

    return img