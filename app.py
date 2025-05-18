import os
HOME = os.getcwd()
print("HOME:", HOME)

!git clone https://github.com/facebookresearch/segment-anything-2.git
%cd {HOME}/segment-anything-2
!pip install -e . -q

!pip install -q supervision jupyter_bbox_widget
!pip install ultralytics

!mkdir -p {HOME}/checkpoints
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P {HOME}/checkpoints
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P {HOME}/checkpoints
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P {HOME}/checkpoints
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P {HOME}/checkpoints

import cv2
import torch
import base64

import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"{HOME}/checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

%%writefile points.py


import base64
import cv2
from PIL import Image
from ultralytics import YOLO
from jupyter_bbox_widget import BBoxWidget
import os
HOME = os.getcwd()
print("HOME:", HOME)

def encode_image(filepath):
    """
    Encode the image at the given file path to a base64 string.

    Args:
        filepath (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64," + encoded

def load_and_preprocess_image(image_path):
    """
    Load an image from the given path and convert it to RGB format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The RGB image.
    """
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def get_bounding_box_coordinates(image_path, model_path="/kaggle/input/trainedmodel/best.pt"):
    """
    Get the coordinates of the bounding box with the highest confidence score
    from the given image using a YOLO model.

    Args:
        image_path (str): The path to the image file.
        model_path (str): The path to the YOLO model file.

    Returns:
        list: A list of dictionaries containing the bounding box coordinates.
    """
    # Load a pretrained YOLO model
    model = YOLO(model_path)

    # Run inference on images
    results = model([image_path])

    # Get the bounding box with the highest confidence score
    for r in results:
        best_box = r.boxes[r.boxes.conf.argmax()]  # Assuming 'conf' is confidence

        # Extract coordinates of the best box
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        return [
            {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1, 'label': ''}
        ]

def main(image_path):
    """
    Main function to process the image and get the bounding box coordinates.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list: A list of dictionaries containing the bounding box coordinates.
    """
    # Load and preprocess the image
    image_rgb = load_and_preprocess_image(image_path)

    # Encode the image for the widget
    encoded_image = encode_image(image_path)

    # Initialize the widget
    widget = BBoxWidget()
    widget.image = encoded_image

    # Get bounding box coordinates
    bounding_boxes = get_bounding_box_coordinates(image_path)

    # Set the bounding boxes in the widget
    widget.bboxes = bounding_boxes
    return widget.bboxes

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process an image and get bounding box coordinates.")
    parser.add_argument('image_path', type=str, help="The path to the image file")

    args = parser.parse_args()
    bboxes = main(args.image_path)
    print("Bounding Boxes:", bboxes)

!pip install flask flask_cors pyngrok

from pyngrok import ngrok
ngrok.set_auth_token("your_pyngrok_auth_token")

!pip install cloudinary

from flask import Flask, jsonify, request
import requests
import os
import base64
from flask_cors import CORS
from points import main
from cloudinary.uploader import upload as cloudinary_upload
from cloudinary.utils import cloudinary_url
import cv2
import numpy as np
predictor = SAM2ImagePredictor(sam2_model)
from jupyter_bbox_widget import BBoxWidget
widget = BBoxWidget()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Cloudinary configuration
import cloudinary
cloudinary.config(
    cloud_name='dhsm2edbl',  # Replace with your Cloudinary cloud name
    api_key='373461989947699',        # Replace with your Cloudinary API key
    api_secret='yBSD78MuiRTaS8A8z0qEpFFwOAE'  # Replace with your Cloudinary API secret
)

# Define a route
@app.route('/generate-billboard', methods=['POST'])
def process_billboard():
    print("called")
    """Process billboard image URL and return segmented image URL."""
    data = request.json
    image_url = data.get('billboardUrl')

    if not image_url:
        return jsonify({"error": "No billboard URL provided."}), 400

    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download billboard image."}), 400

        image_path = os.path.join('temp', 'billboard.jpg')
        os.makedirs('temp', exist_ok=True)
        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Process the image
        bboxes = main(image_path)
        print("Bounding Boxes:", bboxes)
        def encode_image(filepath):
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
            encoded = str(base64.b64encode(image_bytes), 'utf-8')
            return "data:image/jpg;base64,"+encoded

        widget.image = encode_image(image_path)
        widget.bboxes = main(image_path)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        """my_boxes = [
           {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1, 'label': ''}
        ]"""
        #widget.bboxes = my_boxes
        widget.bboxes

        default_box = [
            {'x': 166, 'y': 835, 'width': 99, 'height': 175, 'label': ''},
            {'x': 472, 'y': 885, 'width': 168, 'height': 249, 'label': ''},
            {'x': 359, 'y': 727, 'width': 27, 'height': 155, 'label': ''},
            {'x': 164, 'y': 1044, 'width': 279, 'height': 163, 'label': ''}
        ]

        boxes = widget.bboxes if widget.bboxes else default_box
        boxes = np.array([
            [
                box['x'],
                box['y'],
                box['x'] + box['width'],
                box['y'] + box['height']
            ] for box in boxes
        ])

        predictor.set_image(image_rgb)

        masks, scores, logits = predictor.predict(
            box=boxes,
            multimask_output= 1
        )

            # With one box as input, predictor returns masks of shape (1, H, W);
            # with N boxes, it returns (N, 1, H, W).
        if boxes.shape[0] != 1:
            masks = np.squeeze(masks)
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks.astype(bool)
        )

        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)   
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        # Upload images to Cloudinary
        segmented_upload = cloudinary_upload(cv2.imencode('.jpg', segmented_image)[1].tobytes(), resource_type='image')
        
        return jsonify({
            "message": "Billboard processed successfully.",
            "billboardUrl": image_url,
            "segmentedBillboardUrl": segmented_upload['secure_url']
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 500

@app.route('/generate-banner', methods=['POST'])
def process_banner():
    print("runnning")
    """Process banner and billboard image URLs and return the final billboard URL."""
    data = request.json
    billboard_url = data.get('billboardUrl')
    banner_url = data.get('bannerUrl')

    if not billboard_url or not banner_url:
        return jsonify({"error": "Both billboard and banner URLs are required."}), 400

    try:
        # Download images from URLs
        billboard_response = requests.get(billboard_url, stream=True)
        banner_response = requests.get(banner_url, stream=True)

        if billboard_response.status_code != 200 or banner_response.status_code != 200:
            return jsonify({"error": "Failed to download images."}), 400

        image_path = os.path.join('temp', 'billboard.jpg')
        banner_path = os.path.join('temp', 'banner.jpg')
        os.makedirs('temp', exist_ok=True)

        with open(image_path, 'wb') as f:
            f.write(billboard_response.content)

        with open(banner_path, 'wb') as f:
            f.write(banner_response.content)

        # Process the images
        bboxes = main(image_path)
        print("Bounding Boxes:", bboxes)
        def encode_image(filepath):
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
            encoded = str(base64.b64encode(image_bytes), 'utf-8')
            return "data:image/jpg;base64,"+encoded
        print("image_path : ",image_path)
        widget = BBoxWidget()
        widget.image = encode_image(image_path)
            # file_data1 = file1.read()
            # encoded_image1 = str(base64.b64encode(file_data1).decode('utf-8'))
            # widget.image = "data:image/jpg;base64,"+encoded
        widget
        widget.bboxes = main(image_path)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        widget.bboxes

        default_box = [
            {'x': 166, 'y': 835, 'width': 99, 'height': 175, 'label': ''},
            {'x': 472, 'y': 885, 'width': 168, 'height': 249, 'label': ''},
            {'x': 359, 'y': 727, 'width': 27, 'height': 155, 'label': ''},
            {'x': 164, 'y': 1044, 'width': 279, 'height': 163, 'label': ''}
        ]

        boxes = widget.bboxes if widget.bboxes else default_box
        boxes = np.array([
            [
                box['x'],
                box['y'],
                box['x'] + box['width'],
                box['y'] + box['height']
            ] for box in boxes
        ])

        predictor.set_image(image_rgb)

        masks, scores, logits = predictor.predict(
            box=boxes,
            multimask_output= 1
        )

# With one box as input, predictor returns masks of shape (1, H, W);
# with N boxes, it returns (N, 1, H, W).
        if boxes.shape[0] != 1:
            masks = np.squeeze(masks)
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks.astype(bool)
        )

        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        def find_billboard_corners(combined_mask):
            mask_uint8 = (combined_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                 return None  # Ensure the contour has exactly 4 corners

    # Sort the points in a consistent order: top-left, top-right, bottom-right, bottom-left
            approx = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=1)
            diff = np.diff(approx, axis=1)

            rect[0] = approx[np.argmin(s)]  # Top-left
            rect[2] = approx[np.argmax(s)]  # Bottom-right
            rect[1] = approx[np.argmin(diff)]  # Top-right
            rect[3] = approx[np.argmax(diff)]  # Bottom-left

            return rect

        def transform_overlay(image_bgr, overlay_image, corners):
    # Ensure corners are float32 for precision
            src_pts = np.array([[0, 0],
                                [overlay_image.shape[1] - 1, 0],
                                [overlay_image.shape[1] - 1, overlay_image.shape[0] - 1],
                                [0, overlay_image.shape[0] - 1]], dtype="float32")
            dst_pts = np.array(corners, dtype="float32")

    # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the overlay image to fit the detected billboard
            warped_overlay = cv2.warpPerspective(overlay_image, M, (image_bgr.shape[1], image_bgr.shape[0]))

    # Create a mask for blending
            overlay_mask = np.zeros_like(image_bgr, dtype=np.uint8)
            cv2.fillConvexPoly(overlay_mask, corners.astype(int), (255, 255, 255))

    # Combine the warped overlay with the original image
            combined_image = cv2.bitwise_and(image_bgr, cv2.bitwise_not(overlay_mask))
            combined_image = cv2.bitwise_or(combined_image, warped_overlay)

            return combined_image

# Combine all billboard masks into a single binary mask
        combined_mask = np.any(masks, axis=0).astype('uint8')

# Get corners from the combined mask
        corners = find_billboard_corners(combined_mask)
        print("Corners by SAM2:", corners)
        if corners is not None:
    # Read the overlay image
            overlay_image = cv2.imread(banner_path)

    # Transform and overlay the image
            combined_image = transform_overlay(image_bgr, overlay_image, corners)

            # combined_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            combined_upload = cloudinary_upload(cv2.imencode('.jpg', combined_image)[1].tobytes(), resource_type='image')

        return jsonify({
            "message": "Final billboard created successfully.",
            "finalBillboardUrl": combined_upload['secure_url']
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process the images: {str(e)}"}), 500

# Utility function for overlaying banner on billboard
def process_overlay(image_bgr, overlay_image):
    # Implement perspective transformation and overlay logic
    pass

@app.route('/')
def home():
    return jsonify({"home": "Welcome to the backend"})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)  # Expose port 5000 to the public internet
    print("Public URL:", public_url)  
    app.run(host='0.0.0.0', port=5000)