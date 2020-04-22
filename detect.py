
from torchvision import transforms
from utils.utils import *
from PIL import Image, ImageDraw, ImageFont
from model.BlazeFace import BlazeFace
import torch
from utils.dataset import Uplara
from utils.gen_anchors import get_anchors
import cv2
import matplotlib.pyplot as plt


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Move to default device

    # Forward prop.
    # rev_label_map = {'background','left'}
    predicted_scores, predicted_locs = model(original_image.unsqueeze(0))
    # print(predicted_locs.size())
    # print(predicted_scores.size())

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores, get_anchors(), min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    print(det_labels, det_scores)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # Transform to original image dimensions
    # original_dims = torch.FloatTensor(
    #     [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    # det_boxes = det_boxes * original_dims

    # Decode class integer labels
    # det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    # if det_labels == ['background']:
    #     # Just return original image
    #     return original_image

    # Annotate

    image = original_image.permute([1,2,0]).cpu().numpy()
    # draw = ImageDraw.Draw(annotated_image)
    # font = ImageFont.truetype("./calibril.ttf", 15)
    print(det_boxes.size())
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        print(box_location)
        cv2.rectangle(image, (int(box_location[0]), int(box_location[1])), (int(box_location[2]), int(box_location[3])), (0,255,0), 1)
        plt.imshow(image)
        plt.show()
        # draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        # draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        # text_size = font.getsize(det_labels[i].upper())
        # text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        # textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
        #                     box_location[1]]
        # draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
        #           font=font)
    # del draw

    return image


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlazeFace()
    model.load_state_dict(torch.load('/home/noldsoul/Desktop/MyProjects/a-PyTorch-Tutorial-to-Object-Detection/mobilenet_objectdetection.pt', map_location = torch.device(device)))
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_dataset.csv", "/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_images/", transform = transform)
    image, _,_ = dataset[0]
    image = image.to(device)

    detect(image, min_score=0.2, max_overlap=0.5, top_k=10).show()
