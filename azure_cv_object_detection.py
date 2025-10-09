# Azure Custom Vision for object detection
# Serge Retkowsky - Microsoft - serge.retkowsky@microsoft.com
# 09/10/2025

import cv2
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import time

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from datetime import datetime, timedelta
from decimal import Decimal
from dotenv import load_dotenv
from IPython.display import display, FileLink
from msrest.authentication import ApiKeyCredentials
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv("azure.env")

training_endpoint = os.environ["training_endpoint"]
prediction_endpoint = os.environ["prediction_endpoint"]
training_key = os.environ["training_key"]
prediction_key = os.environ["prediction_key"]
prediction_resource_id = os.environ["prediction_resource_id"]

training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(training_endpoint, training_credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)


def normalize_bbox_coordinates(
        row) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Normalize bounding box coordinates to [0, 1] range with proper bounds checking.
    
    Args:
        row: DataFrame row containing bbox and image dimensions
    
    Returns:
        Tuple of (x, y, width, height) as Decimal objects
        
    Raises:
        ValueError: If coordinates are invalid
    """
    try:
        # Normalize coordinates
        x = max(0, min(1, Decimal(row.ann_bbox_xmin) / Decimal(row.img_width)))
        y = max(0, min(1, Decimal(row.ann_bbox_ymin) / Decimal(row.img_height)))

        # Calculate width and height with bounds checking
        max_width = Decimal('1') - x
        max_height = Decimal('1') - y

        w = max(0, min(max_width, Decimal(row.ann_bbox_width) / Decimal(row.img_width)))
        h = max(0, min(max_height, Decimal(row.ann_bbox_height) / Decimal(row.img_height)))

        # Validate non-zero dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox dimensions: width={w}, height={h}")

        return x, y, w, h

    except (ZeroDivisionError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to normalize bbox coordinates: {e}")

def create_regions_for_image(img_df, tags) -> List[Region]:
    """
    Create Region objects for all annotations in an image.
    
    Args:
        img_df: DataFrame containing annotations for a single image
        tags: Dictionary mapping category names to tag objects
    
    Returns:
        List of Region objects
    """
    regions = []

    for _, row in img_df.iterrows():
        try:
            # Check if category exists in tags
            if row.cat_name not in tags:
                logger.warning(f"Category '{row.cat_name}' not found in tags, skipping region")
                continue

            x, y, w, h = normalize_bbox_coordinates(row)

            regions.append(
                Region(tag_id=tags[row.cat_name].id,
                       left=x,
                       top=y,
                       width=w,
                       height=h))

        except ValueError as e:
            logger.warning(f"Skipping region in image {img_df.iloc[0].img_filename}: {e}")
            continue

    return regions

def upload_image_batch(trainer, project_id: str,
                       image_entries: List[ImageFileCreateEntry]) -> bool:
    """
    Upload a batch of images to Azure Custom Vision.
    
    Args:
        trainer: Azure Custom Vision trainer client
        project_id: Project ID
        image_entries: List of image entries to upload
    
    Returns:
        True if upload successful, False otherwise
    """
    try:
        upload_result = trainer.create_images_from_files(
            project_id, ImageFileCreateBatch(images=image_entries))

        if not upload_result.is_batch_successful:
            logger.error("Batch upload failed:")
            for i, image_result in enumerate(upload_result.images):
                if image_result.status != "OK":
                    logger.error(f"  Image {i}: {image_result.status}")
            return False

        return True

    except Exception as e:
        logger.error(f"Exception during batch upload: {e}")
        return False

def upload_images_to_azure_custom_vision(
  dataset: Any,
  trainer: CustomVisionTrainingClient,
  project: str,
  tags: str,
  project_name: str,
  batch_size: int = 64
) -> Tuple[int, int]:
    """
    Upload images and labels to Azure Custom Vision project with improved error handling and progress tracking.
    
    Args:
        dataset: Dataset object containing images and annotations
        trainer: Azure Custom Vision trainer client
        project: Azure Custom Vision project object
        tags: Dictionary mapping category names to tag objects
        project_name: Name of the project for logging
        batch_size: Number of images to upload in each batch (max 64 for Azure)
    """
    start_time = time.time()
    successful_uploads = 0
    failed_uploads = 0
    total_images = len(dataset.df.groupby('img_filename'))

    logger.info(f"Starting upload of {total_images} images to Azure Custom Vision project: '{project_name}'")

    # Group images for batch processing
    image_groups = list(dataset.df.groupby('img_filename'))
    image_entries = []

    for i, (img_filename, img_df) in enumerate(image_groups, 1):
        # Construct image path using pathlib for better cross-platform compatibility
        img_folder = img_df.iloc[0].img_folder
        img_path = Path(dataset.path_to_annotations) / str(img_folder) / img_filename

        # Check if image file exists
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            failed_uploads += 1
            continue

        # Create regions for this image
        regions = create_regions_for_image(img_df, tags)

        if not regions:
            logger.warning(f"No valid regions found for image: {img_filename}")
            failed_uploads += 1
            continue

        # Read image file
        try:
            with open(img_path, "rb") as image_file:
                image_entry = ImageFileCreateEntry(name=img_filename,
                                                   contents=image_file.read(),
                                                   regions=regions)
                image_entries.append(image_entry)

        except (IOError, OSError) as e:
            logger.error(f"Failed to read image file {img_path}: {e}")
            failed_uploads += 1
            continue

        # Upload batch when it reaches batch_size or is the last image
        if len(image_entries) >= batch_size or i == len(image_groups):
            if upload_image_batch(trainer, project.id, image_entries):
                successful_uploads += len(image_entries)
                logger.info(
                    f"Successfully uploaded batch of {len(image_entries)} images "
                    f"({successful_uploads}/{total_images} total)")
            else:
                failed_uploads += len(image_entries)
                logger.error(
                    f"Failed to upload batch of {len(image_entries)} images")

            # Clear batch
            image_entries = []

        # Progress update every 10% or 50 images, whichever is smaller
        progress_interval = min(50, max(1, total_images // 10))
        if i % progress_interval == 0 or i == total_images:
            progress_percent = (i / total_images) * 100
            logger.info(f"Progress: {i}/{total_images} images processed ({progress_percent:.1f}%)")

    # Final results
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    logger.info(f"\n{'='*50}")
    logger.info(f"Upload Summary:")
    logger.info(f"  Total images processed: {total_images}")
    logger.info(f"  Successful uploads: {successful_uploads}")
    logger.info(f"  Failed uploads: {failed_uploads}")
    logger.info(f"  Success rate: {(successful_uploads/total_images)*100:.1f}%"
                if total_images > 0 else "N/A")
    logger.info(
        f"  Elapsed time: {minutes:.0f} minutes and {seconds:.0f} seconds")
    logger.info(f"{'='*50}")

    return successful_uploads, failed_uploads

def get_iteration_performance(
  trainer: CustomVisionTrainingClient,
  project_id: str,
  iteration_id: str,
  threshold: float = 0.5,
  overlap: float = 0.3,
):
    """
    Get comprehensive performance metrics for a trained Custom Vision iteration.
    
    Args:
        trainer: CustomVisionTrainingClient instance
        project_id: Project ID
        iteration_id: Iteration ID to evaluate
        threshold: Probability threshold for predictions (0.0-1.0)
    Returns:
        dict: Performance metrics and details
    """
    try:
        print(f"📊 Retrieving performance metrics for iteration: {iteration_id}")
        print("=" * 60)

        # Get iteration details
        iteration = trainer.get_iteration(project_id, iteration_id)
        print(f"📋 Iteration Name: {iteration.name}")
        print(f"📅 Created: {iteration.created}")
        print(f"🔄 Status: {iteration.status}")
        print(f"⚙️  Training Type: {iteration.training_type}")

        if iteration.status != "Completed":
            print(f"⚠️  Warning: Iteration status is '{iteration.status}', not 'Completed'")

        # Get performance metrics
        performance = trainer.get_iteration_performance(
            project_id, iteration_id, threshold, overlap)

        print("\n🏆 OVERALL PERFORMANCE METRICS")
        print("=" * 40)
        print(f"📈 Precision: {performance.precision:.4f} ({performance.precision*100:.2f}%)")
        print(f"📈 Recall: {performance.recall:.4f} ({performance.recall*100:.2f}%)")
        print(f"📈 Average Precision (AP): {performance.average_precision:.4f} ({performance.average_precision*100:.2f}%)")

        # Per-tag performance
        print(f"\n🏷️  PER-TAG PERFORMANCE (Threshold: {threshold})")
        print("=" * 50)

        tag_metrics = []
        for tag_perf in performance.per_tag_performance:
            tag_data = {
                'tag_name': tag_perf.name,
                'tag_id': tag_perf.id,
                'precision': tag_perf.precision,
                'recall': tag_perf.recall,
                'average_precision': tag_perf.average_precision
            }
            tag_metrics.append(tag_data)

            print(f"🏷️  Tag: {tag_perf.name}")
            print(f"   📊 Precision: {tag_perf.precision:.4f} ({tag_perf.precision*100:.2f}%)")
            print(f"   📊 Recall: {tag_perf.recall:.4f} ({tag_perf.recall*100:.2f}%)")
            print(f"   📊 Average Precision: {tag_perf.average_precision:.4f} ({tag_perf.average_precision*100:.2f}%)")

        # Performance interpretation
        print(f"\n💡 PERFORMANCE INTERPRETATION")
        print("=" * 35)

        overall_score = (performance.precision + performance.recall +
                         performance.average_precision) / 3

        if overall_score >= 0.9:
            rating = "🌟 Excellent"
        elif overall_score >= 0.8:
            rating = "✅ Very Good"
        elif overall_score >= 0.7:
            rating = "👍 Good"
        elif overall_score >= 0.6:
            rating = "⚠️  Fair"
        else:
            rating = "❌ Needs Improvement"

        print(f"Overall Rating: {rating} (Score: {overall_score:.3f})")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 20)

        if performance.precision < 0.7:
            print("📢 Low Precision: Model has many false positives")
            print("   • Add more negative examples")
            print("   • Remove ambiguous training images")
            print("   • Increase probability threshold")

        if performance.recall < 0.7:
            print("📢 Low Recall: Model misses many true positives")
            print("   • Add more diverse training images per tag")
            print("   • Include edge cases and variations")
            print("   • Lower probability threshold")

        if performance.average_precision < 0.8:
            print("📢 Low Average Precision: Overall model confidence is low")
            print("   • Try Advanced Training mode")
            print("   • Increase training budget")
            print("   • Add higher quality training images")

        # Return structured data
        return {
            'iteration_id': iteration_id,
            'iteration_name': iteration.name,
            'status': iteration.status,
            'training_type': iteration.training_type,
            'created': iteration.created,
            'threshold': threshold,
            'overall_metrics': {
                'precision': performance.precision,
                'recall': performance.recall,
                'average_precision': performance.average_precision,
                'overall_score': overall_score
            },
            'per_tag_metrics': tag_metrics
        }

    except Exception as e:
        print(f"❌ Error getting performance metrics: {str(e)}")
        return None

def compare_iterations(trainer, project_id: str, iteration_ids, threshold: float = 0.5):
    """
    Compare performance metrics across multiple iterations.
    
    Args:
        trainer: CustomVisionTrainingClient instance
        project_id: Project ID
        iteration_ids: List of iteration IDs to compare
        threshold: Probability threshold for predictions
    Returns:
        dict: Comparison results
    """
    print(f"⚖️  COMPARING {len(iteration_ids)} ITERATIONS")
    print("=" * 50)

    results = []

    for i, iteration_id in enumerate(iteration_ids, 1):
        print(f"\n📊 Iteration {i}: {iteration_id}")
        print("-" * 30)

        metrics = get_iteration_performance(trainer, project_id, iteration_id,
                                            threshold)
        if metrics:
            results.append(metrics)

    if len(results) > 1:
        print(f"\n🏆 COMPARISON SUMMARY")
        print("=" * 25)

        best_precision = max(results,
                             key=lambda x: x['overall_metrics']['precision'])
        best_recall = max(results,
                          key=lambda x: x['overall_metrics']['recall'])
        best_ap = max(results,
                      key=lambda x: x['overall_metrics']['average_precision'])
        best_overall = max(results,
                           key=lambda x: x['overall_metrics']['overall_score'])

        print(f"🥇 Best Precision: {best_precision['iteration_name']} ({best_precision['overall_metrics']['precision']:.3f})")
        print(f"🥇 Best Recall: {best_recall['iteration_name']} ({best_recall['overall_metrics']['recall']:.3f})")
        print(f"🥇 Best Average Precision: {best_ap['iteration_name']} ({best_ap['overall_metrics']['average_precision']:.3f})")
        print(f"🏆 Best Overall: {best_overall['iteration_name']} (Score: {best_overall['overall_metrics']['overall_score']:.3f})")

    return results

def export_metrics_to_json(metrics: dict, filename: str = None):
    """
    Exports the given metrics dictionary to a JSON file.

    Parameters:
    ----------
    metrics : dict
        A dictionary containing the metrics to be exported.
    filename : str, optional
        The name of the output JSON file. If not provided, a filename will be
        generated using the current timestamp in the format:
        'custom_vision_metrics_YYYYMMDD_HHMMSS.json'.

    Returns:
    -------
    str or None
        The name of the file the metrics were exported to, or None if an error occurred.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_vision_metrics_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"📄 Metrics exported to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error exporting metrics: {str(e)}")
        return None

def get_project_id_by_name(project_name: str):
    """
    Retrieves the ID of a project by its name using the Custom Vision trainer.

    Parameters:
    ----------
    project_name : str
        The name of the project to search for.

    Returns:
    -------
    str or None
        The ID of the project if found, otherwise None.
    """
    projects = trainer.get_projects()

    for project in projects:
        if project.name.lower() == project_name.lower():
            return project.id

    print(f"❌ Project '{project_name}' not found!")
    return None

def get_specific_project_info(project_id: str):
    """
    Displays detailed information about a specific Custom Vision project and its iterations.

    Parameters:
    ----------
    project_id : str
        The unique identifier of the Custom Vision project.
    """
    project = trainer.get_project(project_id)
    iteration_perf = trainer.get_iterations(project_id)

    print(f"🗂️ Project: {project.name}")
    print(f"🆔 Project id: {project_id}")
    print(f"📊 Total iterations = {len(iteration_perf)}")

    for iteration in iteration_perf:
        status_emoji = {
            "Training": "🏋️",
            "Completed": "✅",
            "Error": "❌",
            "Warning": "⚠️",
            "Pending": "⏳",
        }.get(iteration.status, "ℹ️")

        print(f"  🌀 Iteration: {iteration.name}")
        print(f"  {status_emoji} Status: {iteration.status}")
        print(f"  📅 Created: {iteration.created}")

def get_tag_statistics(project_id: str):
    """
    Retrieves detailed statistics about tags in a Custom Vision project.

    Parameters:
    ----------
    project_id : str
        The unique identifier of the Custom Vision project.

    Returns:
    -------
    dict
    """
    tags = trainer.get_tags(project_id)

    stats = {
        'total_tags': len(tags),
        'tags_with_images': 0,
        'total_tagged_images': 0,
        'tag_details': []
    }

    for tag in tags:
        if tag.image_count > 0:
            stats['tags_with_images'] += 1
            stats['total_tagged_images'] += tag.image_count

        stats['tag_details'].append({
            'name': tag.name,
            'image_count': tag.image_count,
            'id': tag.id
        })

    return stats

def model_training(project_id: str, display_interval: int = 60, timeout: int = 10800):
    """
    Run a quick training on the project and wait for it to complete.

    Args:
        project_id (str): The project ID
        poll_interval (int): Seconds to wait between status checks
        timeout (int): Max time to wait in sec before giving up

    Returns:
        iteration: The completed training iteration object, or None on failure
    """
    print(f"🧠 Starting 'Quick Training'\n")
    start_time = time.time()
  
    try:
        # Start the training
        iteration = trainer.train_project(project_id, training_type="Regular")
                            
        iteration_id = iteration.id
        print(f"🎯 Training started. Iteration ID: {iteration_id}")
        print(f"⏳ Initial Status: {iteration.status}\n")

        # Wait for training to complete
        elapsed_time = 0
        while iteration.status.lower() not in (
                "completed", "failed") and elapsed_time < (timeout):
            time.sleep(display_interval)
            elapsed_time += display_interval  # Refresh the iteration status
            
            iteration = trainer.get_iteration(project_id, iteration_id)  

            now = datetime.today().strftime('%d-%b-%Y %H:%M:%S')
            disp_elapsed_time = time.time() - start_time
            h, r = divmod(disp_elapsed_time, 3600)
            m, s = divmod(r, 60)
            print(f"[{now}] Current Status: {iteration.status} | Elapsed time: {int(h)} hours, {int(m)} minutes, {s:.0f} seconds")

        if iteration.status.lower() == "completed":
            print("\n✅ Training completed successfully.")
            return iteration
        else:
            print(
                f"❌ Training failed or timed out. Final status: {iteration.status}"
            )
            return None

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None
     
def get_predictions(project_id: str, publish_iteration_name: str, image_path: str, output_dir: str, min_pred: float = 0.7):
    """
    Performs object detection on an image using a Custom Vision model and returns predictions above a confidence threshold.

    Parameters:
    ----------
    project_id: str Project ID
    publish_iteration_name: str Iteration name of the deployed model
    image_path: str Path to the input image file.
    output_dir: Directory to save the output images
    min_pred : float, optional  Minimum confidence threshold for predictions to be included (default is 0.7).

    Returns:
    -------
    tuple
        A tuple containing:
        - output_image_file (str): Path to the saved annotated image.
        - predictions_list (list): A list of dictionaries with prediction details:
            - 'tag': Predicted tag name.
            - 'probability': Confidence score of the prediction.
            - 'bbox': Bounding box coordinates (normalized).
    """
    img = cv2.imread(image_path)
    image_height, image_width = img.shape[:2]

    with open(image_path, mode="rb") as image_data:
        results = predictor.detect_image(project_id, publish_iteration_name,
                                         image_data)

    predictions_list = []
    nb = 0

    print(f"Analyzing {image_path}\n")
    
    for prediction in results.predictions:
        if prediction.probability >= min_pred:
            nb += 1
            tag = prediction.tag_name
            prob = prediction.probability

            # Convert normalized bbox to absolute coordinates
            left = int(prediction.bounding_box.left * image_width)
            top = int(prediction.bounding_box.top * image_height)
            width = int(prediction.bounding_box.width * image_width)
            height = int(prediction.bounding_box.height * image_height)
            right = left + width
            bottom = top + height

            # Store detection info
            detection_info = {
                'tag': tag,
                'probability': prob,
                'bbox': {
                    'left': prediction.bounding_box.left,
                    'top': prediction.bounding_box.top,
                    'width': prediction.bounding_box.width,
                    'height': prediction.bounding_box.height
                }
            }
            predictions_list.append(detection_info)

            # Draw rectangle
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label background
            label = f"{tag} ({prob:.2f})"
            (text_width,
             text_height), baseline = cv2.getTextSize(label,
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.5, 1)
            cv2.rectangle(img, (left, top - text_height - 5),
                          (left + text_width, top), (0, 255, 0), -1)

            # Put label text
            cv2.putText(img, label, (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(
                f"{nb:2} 🎯 Detected {tag} = {prob:.4}\t📦 left = {prediction.bounding_box.left:.2f} ⬆️ top = {prediction.bounding_box.top:.2f} ↔️ width = {prediction.bounding_box.width:.2f} ↕️ height = {prediction.bounding_box.height:.2f}"
            )

    # Saving the image
    output_image_file = os.path.join(output_dir, f"pred_{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
    cv2.imwrite(output_image_file, img)
    print(f"\n✅ Annotated image saved to {output_image_file}")

    return output_image_file, predictions_list
  