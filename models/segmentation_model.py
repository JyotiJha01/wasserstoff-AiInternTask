# import torch
# import torchvision
# from utils.postprocessing import extract_and_save_objects
# from utils.postprocessing import extract_and_save_objects
# from .identification_model import IdentificationModel
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.transforms import functional as F
# from PIL import Image
# import numpy as np
# import os

# class SegmentationModel:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = maskrcnn_resnet50_fpn(pretrained=True)
#         self.identification_model = IdentificationModel()
#         self.model.to(self.device)
#         self.model.eval()

#     def segment_image(self, image_path):
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             prediction = self.model(image_tensor)[0]

#         masks = prediction['masks']
#         scores = prediction['scores']
#         labels = prediction['labels']

#         confidence_threshold = 0.7
#         high_confidence_masks = masks[scores > confidence_threshold]

#         segmented_objects = high_confidence_masks.cpu().numpy()
#         return segmented_objects, image

#     def visualize_segmentation(self, image, segmented_objects):
#         color_map = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
#         mask_overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)

#         for i, mask in enumerate(segmented_objects):
#             color = color_map[i % 256]
#             mask = mask.squeeze()
#             mask_overlay[mask > 0.5] = color

#         alpha = 0.5
#         blended = Image.fromarray((np.array(image) * (1 - alpha) + mask_overlay * alpha).astype(np.uint8))
#         return blended

 
    
#     def process_image(self, image_path, output_dir, db_path):
#         segmented_objects, original_image = self.segment_image(image_path)
#         extracted_objects = extract_and_save_objects(segmented_objects, original_image, image_path, output_dir, db_path)
        
#         # Identify objects
#         for obj in extracted_objects:
#             object_path = os.path.join(output_dir, obj['filename'])
#             category, confidence = self.identification_model.identify_object(object_path)
#             obj['category'] = category
#             obj['confidence'] = confidence

#         visualized_image = self.visualize_segmentation(original_image, segmented_objects)
#         return extracted_objects, visualized_image

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os
from utils.postprocessing import extract_and_save_objects
from .identification_model import IdentificationModel

class SegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.identification_model = IdentificationModel()
        self.model.to(self.device)
        self.model.eval()

    def segment_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image_tensor)[0]

        masks = prediction['masks']
        scores = prediction['scores']
        labels = prediction['labels']
        boxes = prediction['boxes']

        confidence_threshold = 0.7
        high_confidence_indices = scores > confidence_threshold
        high_confidence_masks = masks[high_confidence_indices]
        high_confidence_boxes = boxes[high_confidence_indices]

        segmented_objects = []
        for i, mask in enumerate(high_confidence_masks):
            bbox = high_confidence_boxes[i].cpu().numpy().tolist()
            mask = mask.squeeze().cpu().numpy()
            segmented_objects.append({'bbox': bbox, 'mask': mask})

        return segmented_objects, image

    def visualize_segmentation(self, image, segmented_objects):
        color_map = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
        mask_overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)

        for i, obj in enumerate(segmented_objects):
            color = color_map[i % 256]
            mask = obj['mask']
            mask_overlay[mask > 0.5] = color

        alpha = 0.5
        blended = Image.fromarray((np.array(image) * (1 - alpha) + mask_overlay * alpha).astype(np.uint8))
        return blended

    def process_image(self, image_path, output_dir, db_path):
        segmented_objects, original_image = self.segment_image(image_path)
        extracted_objects = extract_and_save_objects(segmented_objects, original_image, image_path, output_dir, db_path)

        for obj in extracted_objects:
            object_path = os.path.join(output_dir, obj['filename'])
            category, confidence = self.identification_model.identify_object(object_path)
            obj['category'] = category
            obj['confidence'] = confidence

        visualized_image = self.visualize_segmentation(original_image, segmented_objects)
        return extracted_objects, visualized_image
