
# import os
# import uuid
# import json
# import sqlite3
# from PIL import Image
# import numpy as np

# def extract_and_save_objects(segmented_objects, original_image, input_image_path, output_dir, db_path):
#     """
#     Extract each segmented object, save it as a separate image, and store metadata in SQLite database.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Ensure the directory for the database exists
#     db_dir = os.path.dirname(db_path)
#     if not os.path.exists(db_dir):
#         os.makedirs(db_dir)

#     # Generate a master ID for the original image
#     master_id = str(uuid.uuid4())

#     # Connect to SQLite database
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # Create table if it doesn't exist
#         cursor.execute('''CREATE TABLE IF NOT EXISTS objects
#                           (id TEXT PRIMARY KEY, master_id TEXT, filename TEXT, bbox TEXT)''')

#         extracted_objects = []

#         for i, mask in enumerate(segmented_objects):
#             # Generate a unique ID for the object
#             object_id = str(uuid.uuid4())

#             # Extract the object using the mask
#             mask = mask.squeeze()
#             object_image = np.array(original_image) * mask[..., np.newaxis]
#             object_image = Image.fromarray(object_image.astype(np.uint8))

#             # Calculate bounding box
#             bbox = mask.nonzero()
#             if bbox[0].size > 0 and bbox[1].size > 0:
#                 min_y, max_y = int(bbox[0].min()), int(bbox[0].max())
#                 min_x, max_x = int(bbox[1].min()), int(bbox[1].max())
#                 bbox = (min_x, min_y, max_x, max_y)
#             else:
#                 continue  # Skip this object if the bounding box is empty

#             # Crop the object image
#             object_image = object_image.crop(bbox)

#             # Save the object image
#             object_filename = f"{object_id}.png"
#             object_path = os.path.join(output_dir, object_filename)
#             object_image.save(object_path)

#             # Store metadata in the database
#             cursor.execute("INSERT INTO objects (id, master_id, filename, bbox) VALUES (?, ?, ?, ?)",
#                            (object_id, master_id, object_filename, json.dumps(bbox)))

#             extracted_objects.append({
#                 'id': object_id,
#                 'master_id': master_id,
#                 'filename': object_filename,
#                 'bbox': bbox
#             })

#         # Commit changes and close the database connection
#         conn.commit()
#         conn.close()

#         return extracted_objects
#     except sqlite3.OperationalError as e:
#         print(f"Error accessing the database: {e}")
#         print(f"Please ensure you have write permissions for the directory: {db_dir}")
#         return []
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return []

# def save_visualization(visualized_image, output_path):
#     """
#     Save the visualization of segmented objects.
#     """
#     visualized_image.save(output_path)

import logging
import os
import uuid
import json
import sqlite3
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_and_save_objects(segmented_objects, original_image, input_image_path, output_dir, db_path):
    """
    Extract each segmented object, save it as a separate image, and store metadata in SQLite database.
    """
    logger.debug(f"Starting extraction with {len(segmented_objects)} segmented objects")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the directory for the database exists
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Generate a master ID for the original image
    master_id = str(uuid.uuid4())

    # Connect to SQLite database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS objects
                  (id TEXT PRIMARY KEY, master_id TEXT, filename TEXT, bbox TEXT, 
                  category TEXT, confidence REAL)''')

        extracted_objects = []

        for i, mask in enumerate(segmented_objects):
            logger.debug(f"Processing object {i+1}")

            # Generate a unique ID for the object
            object_id = str(uuid.uuid4())

            # Extract the object using the mask
            mask = mask.squeeze()
            logger.debug(f"Mask shape: {mask.shape}")
            logger.debug(f"Mask unique values: {np.unique(mask)}")

            object_image = np.array(original_image) * mask[..., np.newaxis]
            object_image = Image.fromarray(object_image.astype(np.uint8))

            # Calculate bounding box
            bbox = mask.nonzero()
            logger.debug(f"Bounding box coordinates: {bbox}")

            if bbox[0].size > 0 and bbox[1].size > 0:
                min_y, max_y = int(bbox[0].min()), int(bbox[0].max())
                min_x, max_x = int(bbox[1].min()), int(bbox[1].max())
                bbox = (min_x, min_y, max_x, max_y)
            else:
                logger.warning(f"No clear bounding box for object {i+1}, using full image")
                bbox = (0, 0, original_image.width, original_image.height)

            logger.debug(f"Final bounding box: {bbox}")

            # Crop the object image
            object_image = object_image.crop(bbox)

            # Save the object image
            object_filename = f"{object_id}.png"
            object_path = os.path.join(output_dir, object_filename)
            object_image.save(object_path)
            logger.debug(f"Saved object image to {object_path}")

            # Store metadata in the database
            cursor.execute("""INSERT INTO objects 
                  (id, master_id, filename, bbox, category, confidence) 
                  VALUES (?, ?, ?, ?, ?, ?)""",
               (object_id, master_id, object_filename, json.dumps(bbox), None, None))

            extracted_objects.append({
                'id': object_id,
                'master_id': master_id,
                'filename': object_filename,
                'bbox': bbox
            })

        # Commit changes and close the database connection
        conn.commit()
        conn.close()

        logger.debug(f"Extracted {len(extracted_objects)} objects")
        return extracted_objects
    except sqlite3.OperationalError as e:
        logger.error(f"Error accessing the database: {e}")
        logger.error(f"Please ensure you have write permissions for the directory: {os.path.dirname(db_path)}")
        raise  # Re-raise the exception instead of returning an empty list

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise  # Re-raise the exception

def save_visualization(visualized_image, output_path):
    """
    Save the visualization of segmented objects.
    """
    visualized_image.save(output_path)

def update_object_metadata(db_path, object_id, category, confidence):
    """
    Update the category and confidence for an object in the database.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""UPDATE objects 
                          SET category = ?, confidence = ?
                          WHERE id = ?""",
                       (category, confidence, object_id))

        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Error accessing the database: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")