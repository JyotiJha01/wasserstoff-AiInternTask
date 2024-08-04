# import unittest
# import os
# import sys
# import random
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.segmentation_model import SegmentationModel
# from utils.preprocessing import preprocess_image, get_image_paths
# from utils.postprocessing import save_segmented_objects, save_visualization

# class TestSegmentation(unittest.TestCase):
#     def setUp(self):
#         self.model = SegmentationModel()
#         self.input_dir = "../data/input_images"
#         self.output_dir = "../data/output"
#         self.visualization_dir = "../data/segmented_objects"

#     def test_segmentation(self):
#         # Get all image paths from the input directory
#         image_paths = get_image_paths(self.input_dir)
        
#         # Ensure there are images in the input directory
#         self.assertTrue(len(image_paths) > 0, f"No images found in the input directory: {self.input_dir}")

#         # Randomly select an image for testing
#         test_image_path = random.choice(image_paths)

#         # Ensure the test image exists
#         self.assertTrue(os.path.isfile(test_image_path), f"Selected test image is not a file: {test_image_path}")

#         # Preprocess the image
#         image_array, original_image = preprocess_image(test_image_path)

#         # Perform segmentation
#         segmented_objects, _ = self.model.segment_image(test_image_path)

#         # Check if segmentation produced results
#         self.assertGreater(len(segmented_objects), 0, "Segmentation did not produce any results")

#         # Save segmented objects
#         save_segmented_objects(segmented_objects, original_image, self.output_dir)

#         # Check if segmented objects were saved
#         self.assertTrue(os.path.exists(self.output_dir), "Output directory for segmented objects does not exist")
#         self.assertGreater(len(os.listdir(self.output_dir)), 0, "No segmented objects were saved")

#         # Visualize segmentation
#         visualized_image = self.model.visualize_segmentation(original_image, segmented_objects)

#         # Save visualization
#         os.makedirs(self.visualization_dir, exist_ok=True)
#         visualization_path = os.path.join(self.visualization_dir, f"segmented_{os.path.basename(test_image_path)}")
#         save_visualization(visualized_image, visualization_path)

#         # Check if visualization was saved
#         self.assertTrue(os.path.exists(visualization_path), f"Visualization was not saved at {visualization_path}")

# if __name__ == '__main__':
#     unittest.main()


import unittest
import os
import sys
import random
import sqlite3
import logging
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from utils.preprocessing import get_image_paths
from utils.postprocessing import save_visualization
from utils.data_mapping import generate_object_descriptions

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test environment")
        self.segmentation_model = SegmentationModel()
        self.identification_model = IdentificationModel()
        self.input_dir = "../data/input_images"
        self.output_dir = "../data/output"
        self.visualization_dir = "../data/segmented_objects"
        self.db_path = "../data/object_metadata.db"

        for directory in [self.input_dir, self.output_dir, self.visualization_dir, os.path.dirname(self.db_path)]:
            os.makedirs(directory, exist_ok=True)

        self.initialize_database()

        logger.info("Test environment set up completed")

    def initialize_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS objects")
            cursor.execute('''CREATE TABLE objects
                            (id TEXT PRIMARY KEY,
                            master_id TEXT,
                            filename TEXT,
                            bbox TEXT,
                            category TEXT,
                            confidence REAL)''')
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully with updated schema")
        except sqlite3.OperationalError as e:
            logger.error(f"Error initializing the database: {e}")
            raise

    def test_segmentation_and_extraction(self):
        logger.info("Starting segmentation and extraction test")

        image_paths = get_image_paths(self.input_dir)
        logger.info(f"Found {len(image_paths)} images in the input directory")
        
        self.assertTrue(len(image_paths) > 0, f"No images found in the input directory: {self.input_dir}")

        test_image_path = random.choice(image_paths)
        logger.info(f"Testing image: {test_image_path}")

        segmented_objects, visualized_image = self.segmentation_model.segment_image(test_image_path)

        for obj in segmented_objects:
            self.assertIn('bbox', obj, "Object bounding box not found")
            self.assertIn('mask', obj, "Segmentation mask not found")

        logger.info("Segmentation and extraction test completed successfully")

if __name__ == '__main__':
    unittest.main()
