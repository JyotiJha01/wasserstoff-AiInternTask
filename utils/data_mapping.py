import sqlite3
import json
import os

def get_object_metadata(db_path, object_id=None, master_id=None):
    """
    Retrieve object metadata from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if object_id:
        cursor.execute("SELECT * FROM objects WHERE id = ?", (object_id,))
    elif master_id:
        cursor.execute("SELECT * FROM objects WHERE master_id = ?", (master_id,))
    else:
        cursor.execute("SELECT * FROM objects")

    results = cursor.fetchall()
    conn.close()

    return results

def update_object_metadata(db_path, object_id, updates):
    """
    Update object metadata in the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    update_query = "UPDATE objects SET "
    update_query += ", ".join([f"{key} = ?" for key in updates.keys()])
    update_query += " WHERE id = ?"

    values = list(updates.values()) + [object_id]
    
    cursor.execute(update_query, values)
    conn.commit()
    conn.close()

import logging

logger = logging.getLogger(__name__)

def generate_object_descriptions(db_path, output_file):
    logger.info(f"Generating object descriptions from database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT master_id, filename, object_categories, confidence FROM objects")
    results = cursor.fetchall()
    
    logger.info(f"Found {len(results)} objects in the database")
    
    descriptions = {}
    for master_id, filename, object_categories, confidence in results:
        if master_id not in descriptions:
            descriptions[master_id] = []
        
        # Handle case where confidence might be None
        if confidence is not None:
            confidence_str = f"{confidence:.2f}"
        else:
            confidence_str = "N/A"
            logger.warning(f"Confidence is None for object in {filename}")
        
        descriptions[master_id].append(f"Object in {filename}: {object_categories} (confidence: {confidence_str})")
    
    conn.close()
    
    with open(output_file, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    logger.info(f"Generated descriptions for {len(descriptions)} images")
    return descriptions

def get_image_descriptions(db_path, master_id):
    """
    Get descriptions for all objects in a specific image.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT filename, object_categories, confidence FROM objects WHERE master_id = ?", (master_id,))
    results = cursor.fetchall()

    descriptions = []
    for filename, object_categories, confidence in results:
        descriptions.append(f"Object in {filename}: {object_categories} (confidence: {confidence:.2f})")

    conn.close()

    return descriptions
