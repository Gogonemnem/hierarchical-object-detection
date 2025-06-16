#!/usr/bin/env python3
"""
Combined UMAP Embedding Visualization with Detection Examples

This script creates a comprehensive visualization that combines:
1. UMAP projection of prototype embeddings (hulls, skeleton, depth color)
2. Real detection examples overlaid as thumbnails with fallback-level encoding
3. Prediction labels displayed under each detection thumbnail

The fallback levels are encoded as:
0 = leaf correct (solid green border)
1 = parent (dashed orange border)  
2 = grandparent (dash-dot gold border)
3 = sibling (dotted darkorange border)
4 = cousin (dotted plum border)
5 = off-branch (solid red border)

Each detection thumbnail shows:
- Image crop with colored border indicating fallback level
- Prediction label and confidence score displayed below the image

IMPORTANT: Expected UMAP Clustering Behavior
==========================================
When visualizing both prototype and query embeddings together, you will observe
two distinct clusters in UMAP space. This is EXPECTED and CORRECT behavior:

1. PROTOTYPE CLUSTER: Raw prototype embeddings from the model
   - These are the learned class representations stored in the model
   - Used as reference points for distance-based classification

2. QUERY CLUSTER: Post-processed query embeddings 
   - These are detection query features after distance calculation
   - Undergo sigmoid activation to convert distance logits to probabilities
   - This post-processing (sigmoid transformation) causes the clustering separation

The clustering reflects the model's actual inference pipeline where:
- Raw embeddings are compared via distance metrics (Euclidean/Hyperbolic)
- Distance logits are then transformed to probabilities via sigmoid
- This transformation creates the distinct clustering in embedding space

Both embedding types use identical transformation pipelines (expmap0 for hyperbolic
space, identity for Euclidean) via the EmbeddingClassifier's 'prototypes' property.
The clustering is due to the probability transformation, not geometric differences.
"""

import argparse
import colorsys
import os
import pathlib
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import cv2  # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import torch
import umap
from mmengine.fileio import load
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root
from hod.utils.tree import HierarchyNode, HierarchyTree
try:
    import scipy.spatial
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from mmdet.apis import init_detector, inference_detector

# Patch torch.load to disable `weights_only=True` introduced in PyTorch 2.6
# This avoids UnpicklingError when resuming from checkpoints saved with full objects.
# Safe to use ONLY when loading checkpoints from trusted sources (e.g., your own training).
_real_load = torch.load

def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False  # allow loading full objects (e.g., mmengine ConfigDict, HistoryBuffer)
    return _real_load(*args, **kwargs)

torch.load = safe_load

# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class VisualizationConfig:
    """Unified configuration for all visualization parameters."""
    
    # Core parameters
    target_examples: int = 20
    batch_size: int = 50
    max_batches: int = 30
    min_score: float = 0.25
    iou_threshold: float = 0.5
    random_state: int = 42
    
    # Projection method
    use_mds: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_spread: float = 1.0
    joint_n_neighbors: int = 10
    joint_min_dist: float = 0.05
    
    # PCA preprocessing
    apply_pca: bool = True
    pca_components: int = 50
    min_samples_for_pca: int = 50
    
    # Validation parameters
    validation_sample_size: int = 10
    min_confidence_score: float = 0.3
    diversity_check_threshold: int = 60
    high_conf_threshold: int = 200


@dataclass
class ValidationResult:
    """Results from nearest neighbor validation."""
    
    space_name: str
    sample_size: int
    matches: int
    match_rate: float
    detailed_results: List[Dict[str, Any]] # Will now include names for mismatches
    
    def print_summary(self):
        """Print validation summary, including mismatch details."""
        print(f"\n{self.space_name} space validation: Nearest neighbor = Prediction")
        print(f"Prediction accuracy via nearest neighbor: {self.matches}/{self.sample_size} ({self.match_rate:.1%})")
        
        if self.match_rate >= 0.9:
            print("✅ Excellent: Nearest neighbor matches prediction")
        elif self.match_rate >= 0.7:
            print("⚠️ Good: Most nearest neighbors match predictions")
        elif self.match_rate >= 0.5:
            print("⚠️ Moderate: Some nearest neighbors match predictions")
        else:
            print("❌ Poor: Nearest neighbors often don't match predictions")

        # Print details for mismatches
        if self.sample_size > 0 and self.matches < self.sample_size:
            print("Mismatch Details (Predicted vs. Nearest Neighbor in this space):")
            for res in self.detailed_results:
                if not res.get('matches', True): # If 'matches' is False or missing (though it should be there)
                    pred_name = res.get('predicted_class_name', f"Idx {res['prediction']}")
                    nn_name = res.get('nn_class_name', f"Idx {res['nearest_neighbor']}")
                    print(f"  - Query {res['query_idx']}: Model Predicted: '{pred_name}', but NN was: '{nn_name}' (Dist: {res.get('distance', 'N/A'):.4f})")

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_last_classification_branch_index_from_state(state_dict: Dict[str, Any]) -> int:
    """Determine the index of the last classification branch from state_dict."""
    max_branch_idx = 0
    for key in state_dict.keys():
        if 'bbox_head.cls_branches.' in key:
            try:
                branch_part = key.split('bbox_head.cls_branches.')[1]
                branch_idx = int(branch_part.split('.')[0])
                max_branch_idx = max(max_branch_idx, branch_idx)
            except (IndexError, ValueError):
                continue
    return max_branch_idx


def get_last_classification_branch_index(model) -> int:
    """Determine the index of the last classification branch in the model."""
    for name, module in model.named_modules():
        if hasattr(module, 'num_pred_layer') and 'bbox_head' in name:
            return module.num_pred_layer - 1
    return 0  # Fallback


def get_target_classifier(model):
    """Get the target classifier branch for embedding collection."""
    last_branch_idx = get_last_classification_branch_index(model)
    is_two_stage = getattr(model, 'as_two_stage', False)
    
    if is_two_stage:
        target_idx = last_branch_idx - 1  # Second-to-last decoder layer
        target_classifier = model.bbox_head.cls_branches[target_idx]
    else:
        target_classifier = model.bbox_head.cls_branches[last_branch_idx]
    
    return target_classifier


# =============================================================================
# EMBEDDING COLLECTION
# =============================================================================

class EmbeddingCollector:
    """
    Collects query embeddings during inference using forward hooks.
    
    This class provides scientifically accurate embeddings by capturing the actual
    feature vectors that get compared to prototype embeddings during detection.
    Both prototype and query embeddings come from get_projected_features and are
    in the same transformation space (256-dimensional projected features).
    
    Attributes:
        embeddings (List[np.ndarray]): Collected embedding arrays from hooks
        hook_handles (List[Tuple]): Stored hook handles for cleanup
    """
    
    def __init__(self) -> None:
        """Initialize the collector with empty storage."""
        self.embeddings: List[np.ndarray] = []
        self.hook_handles: List[Tuple[Any, str, Any]] = []
        
    def clear(self) -> None:
        """Clear collected embeddings for next image."""
        self.embeddings.clear()
        
    def register_hooks(self, model) -> bool:
        """
        Register forward hooks on EmbeddingClassifier modules.
        
        Hooks are placed on the second-to-last classification branch to capture
        the actual query embeddings used for final predictions.
        
        Args:
            model: The detection model with classification branches
            
        Returns:
            bool: True if hooks were successfully registered, False otherwise
        """
        hooks_registered = 0
        
        try:
            # Determine target classification branch (second-to-last)
            last_branch_idx = get_last_classification_branch_index(model)
            target_branch_idx = last_branch_idx - 1
            target_branch = f'cls_branches.{target_branch_idx}'
            
            print(f"Targeting classification branch: {target_branch} "
                  f"(branch {target_branch_idx} of {last_branch_idx+1} total - second-to-last)")
            
            # Find and register hooks on target EmbeddingClassifier modules
            for name, module in model.named_modules():
                if self._is_target_module(name, module, target_branch):
                    if self._register_single_hook(name, module):
                        hooks_registered += 1
                        
            print(f"Successfully registered {hooks_registered} hooks")
            return hooks_registered > 0
            
        except Exception as e:
            print(f"Error registering hooks: {e}")
            # Clean up any partial registration
            self.remove_hooks()
            return False
    
    def _is_target_module(self, name: str, module: Any, target_branch: str) -> bool:
        """Check if module is a target EmbeddingClassifier in the right branch."""
        return (hasattr(module, 'get_projected_features') and 
                target_branch in name and 
                'EmbeddingClassifier' in str(type(module)))
    
    def _register_single_hook(self, name: str, module: Any) -> bool:
        """Register a hook on a single EmbeddingClassifier module."""
        try:
            print(f"Registering hook on: {name}")
            
            # Store original method for restoration
            original_method = module.get_projected_features
            
            # Create hooked version that captures embeddings
            def hooked_method(x):
                result = original_method(x)
                
                # Store embeddings if valid tensor
                if hasattr(result, 'shape') and len(result.shape) >= 2:
                    embeddings_batch = result.detach().cpu().numpy()
                    self.embeddings.append(embeddings_batch)
                    
                return result
            
            # Replace method and store for cleanup
            module.get_projected_features = hooked_method
            self.hook_handles.append((module, 'get_projected_features', original_method))
            
            print(f"Successfully registered method hook on {name}")
            return True
            
        except Exception as e:
            print(f"Failed to register hook on {name}: {e}")
            return False
                
    def remove_hooks(self) -> None:
        """
        Remove all registered hooks and restore original methods.
        
        This ensures proper cleanup and prevents memory leaks or interference
        with subsequent model usage.
        """
        restored_count = 0
        
        try:
            for module, method_name, original_method in self.hook_handles:
                try:
                    setattr(module, method_name, original_method)
                    restored_count += 1
                except Exception as e:
                    print(f"Warning: Failed to restore method {method_name}: {e}")
                    
            self.hook_handles.clear()
            
            if restored_count > 0:
                print(f"Successfully restored {restored_count} original methods")
                
        except Exception as e:
            print(f"Error during hook cleanup: {e}")
            # Still clear handles to prevent retry issues
            self.hook_handles.clear()
        
    def get_embeddings_for_detections(self, bbox_index) -> Optional[np.ndarray]:
        """
        Extract embeddings corresponding to specific detections using bbox_index.
        
        Args:
            bbox_index: Tensor or array of indices indicating which queries from the 
                       original query set were selected for final predictions
        
        Returns:
            Optional[np.ndarray]: Query embeddings corresponding to the selected queries,
                                 or None if extraction fails
        """
        if not self.embeddings:
            print("Warning: No embeddings collected")
            return None
            
        try:
            # Consolidate embeddings into single array
            all_embeddings = self._consolidate_embeddings()
            if all_embeddings is None:
                return None
            
            # Convert bbox_index to numpy array if needed
            bbox_indices = self._convert_bbox_index(bbox_index)
            if bbox_indices is None:
                return None
            
            # Validate indices and extract corresponding embeddings
            return self._extract_embeddings_by_indices(all_embeddings, bbox_indices)
            
        except Exception as e:
            print(f"Error extracting embeddings for detections: {e}")
            return None
    
    def _consolidate_embeddings(self) -> Optional[np.ndarray]:
        """Consolidate collected embeddings into a single array."""
        try:
            if len(self.embeddings) == 1:
                all_embeddings = self.embeddings[0]
            else:
                # Concatenate multiple arrays (fallback case)
                all_embeddings = np.concatenate(self.embeddings, axis=0)
                print(f"Concatenated {len(self.embeddings)} embedding arrays")
            
            # Remove batch dimension if present
            if len(all_embeddings.shape) == 3 and all_embeddings.shape[0] == 1:
                all_embeddings = all_embeddings[0]
                
            return all_embeddings
            
        except Exception as e:
            print(f"Error consolidating embeddings: {e}")
            return None
    
    def _convert_bbox_index(self, bbox_index) -> Optional[np.ndarray]:
        """Convert bbox_index to numpy array format."""
        try:
            if hasattr(bbox_index, 'cpu'):
                return bbox_index.cpu().numpy()
            elif hasattr(bbox_index, 'numpy'):
                return bbox_index.numpy()
            else:
                return np.asarray(bbox_index)
                
        except Exception as e:
            print(f"Error converting bbox_index: {e}")
            return None
    
    def _extract_embeddings_by_indices(self, all_embeddings: np.ndarray, 
                                     bbox_indices: np.ndarray) -> Optional[np.ndarray]:
        """Extract embeddings using the provided indices."""
        try:
            # Validate indices range
            max_index = np.max(bbox_indices) if len(bbox_indices) > 0 else -1
            
            if max_index >= len(all_embeddings):
                print(f"Warning: Index out of range - max index {max_index} "
                      f"but only {len(all_embeddings)} embeddings available")
                return None
            
            # Extract and return selected embeddings
            selected_embeddings = all_embeddings[bbox_indices]
            # print(f"Extracted {len(selected_embeddings)} embeddings using bbox_index")
            return selected_embeddings
            
        except Exception as e:
            print(f"Error extracting embeddings by indices: {e}")
            return None

    def get_embeddings_for_detections_fallback(self, num_detections):
        """Fallback method: Extract first N embeddings (old behavior)."""
        if not self.embeddings:
            return None
            
        # Since we only hook the final layer, we should have one embedding array
        if len(self.embeddings) == 1:
            all_embeddings = self.embeddings[0]
        else:
            # Fallback: concatenate if multiple arrays (shouldn't happen with our targeted hook)
            all_embeddings = np.concatenate(self.embeddings, axis=0)
        
        # Flatten if needed (remove batch dimension)
        if len(all_embeddings.shape) == 3 and all_embeddings.shape[0] == 1:
            all_embeddings = all_embeddings[0]  # Remove batch dimension
        
        # Take the first num_detections embeddings
        if len(all_embeddings) >= num_detections:
            result = all_embeddings[:num_detections]
            return result
        else:
            return all_embeddings


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def run_inference_with_hooks(model, dataset, collector: EmbeddingCollector, 
                           config: Optional[VisualizationConfig] = None,
                           hierarchy: Optional[HierarchyTree] = None, 
                           labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Run inference with forward hooks to collect query embeddings.
    
    Processes images in batches until sufficient diversity across all 6 fallback 
    levels is achieved or maximum batch limit is reached. Uses modular design
    with dedicated processors for better error handling and maintainability.
    
    Args:
        model: Detection model with embedding classifier
        dataset: Dataset to process for inference
        collector: EmbeddingCollector instance for capturing query embeddings
        config: Configuration for inference parameters (uses defaults if None)
        hierarchy: Optional hierarchy tree for diversity checking
        labels: Optional class labels for diversity checking
        
    Returns:
        List[Dict[str, Any]]: Results with embeddings for each processed image
        
    Raises:
        RuntimeError: If hook registration fails
    """
    # Validate inputs and use default config if needed
    config = config or VisualizationConfig()
    results_with_embeddings = []
    
    # Register hooks with proper error handling
    if not collector.register_hooks(model):
        raise RuntimeError("Failed to register hooks for embedding collection")
    
    try:
        # Create batch processor with configuration
        processor = BatchProcessor(config, hierarchy, labels)
        
        # Process dataset in batches
        results_with_embeddings = processor.process_dataset(model, dataset, collector)
        
        print(f"Successfully collected {len(results_with_embeddings)} results")
        
    except Exception as e:
        print(f"Error during inference processing: {e}")
        traceback.print_exc()
        
    finally:
        # Critical: Always remove hooks to prevent memory leaks
        try:
            collector.remove_hooks()
        except Exception as cleanup_error:
            print(f"Warning: Error during hook cleanup: {cleanup_error}")
    
    return results_with_embeddings


class BatchProcessor:
    """
    Handles batch processing of images for inference with diversity checking.
    
    This class manages the batch-wise processing of dataset images, monitors
    diversity across hierarchical fallback levels, and implements early stopping
    criteria for efficient data collection.
    
    Attributes:
        config (VisualizationConfig): Configuration parameters for processing
        hierarchy (Optional[HierarchyTree]): Hierarchy tree for diversity checking
        labels (Optional[List[str]]): Class labels for diversity analysis
        diversity_checker (Optional[DiversityChecker]): Checker for fallback level diversity
    """
    
    def __init__(self, config: VisualizationConfig, 
                 hierarchy: Optional[HierarchyTree] = None,
                 labels: Optional[List[str]] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration for processing parameters
            hierarchy: Optional hierarchy tree for diversity checking
            labels: Optional class labels for diversity checking
        """
        self.config = config
        self.hierarchy = hierarchy
        self.labels = labels
        self.diversity_checker = (
            DiversityChecker(hierarchy, labels) 
            if hierarchy and labels 
            else None
        )
        
    def process_dataset(self, model, dataset, collector: EmbeddingCollector) -> List[Dict[str, Any]]:
        """
        Process dataset in batches until diversity criteria are met.
        
        Args:
            model: Detection model for inference
            dataset: Dataset to process
            collector: Embedding collector for query embeddings
            
        Returns:
            List[Dict[str, Any]]: Results with embeddings from processed images
        """
        results_with_embeddings = []
        img_idx = 0
        batch_count = 0
        
        print(f"Starting dataset processing with config: "
              f"target={self.config.target_examples}, "
              f"batch_size={self.config.batch_size}, "
              f"max_batches={self.config.max_batches}")
        
        while batch_count < self.config.max_batches and img_idx < len(dataset):
            batch_end = min(img_idx + self.config.batch_size, len(dataset))
            
            print(f"Processing batch {batch_count + 1}/{self.config.max_batches}, "
                  f"images {img_idx}-{batch_end-1}")
            
            # Process current batch
            batch_results = self._process_batch(model, dataset, collector, img_idx, batch_end)
            results_with_embeddings.extend(batch_results)
            
            # Check stopping criteria
            if self._should_stop_processing(results_with_embeddings, batch_count):
                break
                
            img_idx = batch_end
            batch_count += 1
            
            # Progress reporting
            if batch_count % 3 == 0:
                self._report_progress(results_with_embeddings, batch_count)
                
        print(f"Batch processing completed. Total results: {len(results_with_embeddings)}")
        return results_with_embeddings
    
    def _process_batch(self, model, dataset, collector: EmbeddingCollector, 
                      start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Process a single batch of images with robust error handling.
        
        Args:
            model: Detection model for inference
            dataset: Dataset to process
            collector: Embedding collector for query embeddings
            start_idx: Starting image index (inclusive)
            end_idx: Ending image index (exclusive)
            
        Returns:
            List[Dict[str, Any]]: Results from successfully processed images
        """
        batch_results = []
        errors_count = 0
        
        for current_idx in range(start_idx, end_idx):
            try:
                result_data = self._process_single_image(model, dataset, collector, current_idx)
                if result_data:
                    batch_results.append(result_data)
                    
            except Exception as e:
                errors_count += 1
                print(f"Warning: Error processing image {current_idx}: {e}")
                
                # Stop if too many consecutive errors
                if errors_count > 5:
                    print("Too many errors in batch, skipping remaining images")
                    break
                continue
        
        if batch_results:
            print(f"  Successfully processed {len(batch_results)} images "
                  f"({errors_count} errors)")
                
        return batch_results
    
    def _process_single_image(self, model, dataset, collector: EmbeddingCollector, 
                            img_idx: int) -> Optional[Dict[str, Any]]:
        """
        Process a single image and extract embeddings with enhanced error handling.
        
        Args:
            model: Detection model for inference
            dataset: Dataset to process
            collector: Embedding collector for query embeddings
            img_idx: Index of image to process
            
        Returns:
            Optional[Dict[str, Any]]: Result data if successful, None otherwise
        """
        # Clear previous embeddings
        collector.clear()
        
        try:
            # Get image information
            img_info = dataset.get_data_info(img_idx)
            img_path = img_info['img_path']
            
            # Run inference with gradient disabled for efficiency
            with torch.no_grad():
                result = inference_detector(model, img_path)
                
            # Validate result structure
            if not hasattr(result, 'pred_instances'):
                return None
                
            pred_instances = result.pred_instances
            num_detections = len(pred_instances.bboxes)
            
            if num_detections == 0:
                return None
            
            # Extract and attach query embeddings
            query_embeddings = self._extract_query_embeddings(
                collector, pred_instances, num_detections
            )
            
            if query_embeddings is not None:
                pred_instances.query_embeddings = torch.from_numpy(query_embeddings)
            
            return {
                'image_idx': img_idx,
                'image_path': img_path,
                'result': result,
                'gt_instances': img_info['instances']
            }
            
        except Exception as e:
            print(f"Error processing image {img_idx}: {e}")
            return None
    
    def _extract_query_embeddings(self, collector: EmbeddingCollector, 
                                 pred_instances, num_detections: int) -> Optional[np.ndarray]:
        """
        Extract query embeddings using bbox_index or fallback method.
        
        Args:
            collector: Embedding collector with captured embeddings
            pred_instances: Prediction instances from inference
            num_detections: Number of detections found
            
        Returns:
            Optional[np.ndarray]: Query embeddings if extraction successful
        """
        # Try primary method using bbox_index
        if hasattr(pred_instances, 'bbox_index'):
            bbox_index = pred_instances.bbox_index
            query_embeddings = collector.get_embeddings_for_detections(bbox_index)
            
            if query_embeddings is not None:
                return query_embeddings
            else:
                print("Warning: Could not extract embeddings using bbox_index")
        else:
            print("Warning: bbox_index not available, using fallback method")
            
        # Fallback to simple extraction method
        return collector.get_embeddings_for_detections_fallback(num_detections)
    
    def _should_stop_processing(self, results_with_embeddings: List[Dict[str, Any]], 
                               batch_count: int) -> bool:
        """
        Check if processing should stop based on diversity and count criteria.
        
        Args:
            results_with_embeddings: Results collected so far
            batch_count: Current batch number
            
        Returns:
            bool: True if processing should stop
        """
        total_results = len(results_with_embeddings)
        
        # Don't check early stopping until minimum threshold
        if total_results < self.config.diversity_check_threshold:
            return False
        
        # Use sophisticated diversity checking if available
        if self.diversity_checker is not None:
            if self.diversity_checker.check_diversity(
                results_with_embeddings, 
                self.config.min_score, 
                self.config.iou_threshold
            ):
                print(f"✓ Found sufficient diversity across all fallback levels "
                      f"after {total_results} results")
                return True
        else:
            # Fallback to simple high-confidence count check
            high_conf_count = self._count_high_confidence_detections(results_with_embeddings)
            if high_conf_count >= self.config.high_conf_threshold:
                print(f"✓ Reached target high-confidence detections: "
                      f"{high_conf_count}/{self.config.high_conf_threshold} "
                      f"after {total_results} results")
                return True
                
        return False
    
    def _count_high_confidence_detections(self, results_with_embeddings: List[Dict[str, Any]]) -> int:
        """
        Count high-confidence detections across all results.
        
        Args:
            results_with_embeddings: Results from inference
            
        Returns:
            int: Number of high-confidence detections
        """
        high_conf_count = 0
        
        for result_data in results_with_embeddings:
            result = result_data.get('result')
            if result and hasattr(result, 'pred_instances'):
                scores = result.pred_instances.scores
                high_conf_count += len([s for s in scores if s >= self.config.min_score])
                
        return high_conf_count
    
    def _report_progress(self, results_with_embeddings: List[Dict[str, Any]], 
                        batch_count: int) -> None:
        """
        Report processing progress with statistics.
        
        Args:
            results_with_embeddings: Results collected so far
            batch_count: Current batch number
        """
        total_results = len(results_with_embeddings)
        high_conf = self._count_high_confidence_detections(results_with_embeddings)
        
        print(f"Progress update - Batch {batch_count}: "
              f"{total_results} total results, "
              f"{high_conf} high-confidence detections")
        
        # Show diversity progress if checker available
        if self.diversity_checker and total_results >= 20:
            print("  Diversity status being checked...")
            self.diversity_checker.check_diversity(
                results_with_embeddings, 
                self.config.min_score, 
                self.config.iou_threshold
            )


class DiversityChecker:
    """
    Checks diversity of fallback levels across detection results.
    
    This class analyzes detection results to ensure adequate representation
    across all 6 hierarchical fallback levels for comprehensive visualization.
    It provides sophisticated diversity metrics and early stopping criteria.
    
    Attributes:
        hierarchy (HierarchyTree): Hierarchy tree for relationship analysis
        labels (List[str]): Class labels for mapping predictions
        level_names (List[str]): Human-readable names for fallback levels
    """
    
    def __init__(self, hierarchy: HierarchyTree, labels: List[str]):
        """
        Initialize the diversity checker.
        
        Args:
            hierarchy: Hierarchy tree containing class relationships
            labels: List of class labels for prediction mapping
        """
        self.hierarchy = hierarchy
        self.labels = labels
        self.level_names = [
            'Leaf', 'Parent', 'Grandparent', 
            'Sibling', 'Cousin', 'Off-branch'
        ]
        
    def check_diversity(self, results_with_embeddings: List[Dict[str, Any]], 
                       min_score: float, iou_threshold: float, 
                       min_per_level: int = 2) -> bool:
        """
        Check if we have sufficient diversity across all 6 fallback levels.
        
        Analyzes detection results to determine if there's adequate representation
        across the hierarchical prediction spectrum for meaningful visualization.
        
        Args:
            results_with_embeddings: Results from inference with embeddings
            min_score: Minimum confidence score for including detections
            iou_threshold: IoU threshold for matching predictions to ground truth
            min_per_level: Minimum examples required per fallback level
            
        Returns:
            bool: True if all levels have sufficient examples, False otherwise
        """
        try:
            level_counts = self._count_fallback_levels(
                results_with_embeddings, min_score, iou_threshold
            )
            
            # Analyze coverage
            sufficient_levels = sum(1 for count in level_counts if count >= min_per_level)
            all_levels_covered = sufficient_levels == 6
            
            # Report detailed status
            print(f"Fallback level diversity: {sufficient_levels}/6 levels "
                  f"with {min_per_level}+ examples")
            
            for name, count in zip(self.level_names, level_counts):
                status = "✓" if count >= min_per_level else "✗"
                if count > 0:
                    print(f"  {status} {name}: {count} examples")
                    
            return all_levels_covered
            
        except Exception as e:
            print(f"Error checking diversity: {e}")
            return False
    
    def _count_fallback_levels(self, results_with_embeddings: List[Dict[str, Any]], 
                              min_score: float, iou_threshold: float) -> List[int]:
        """
        Count examples for each fallback level with robust error handling.
        
        Processes detection results to categorize predictions into hierarchical
        fallback levels based on relationship to ground truth.
        
        Args:
            results_with_embeddings: Results from inference
            min_score: Minimum confidence score for including detections  
            iou_threshold: IoU threshold for matching to ground truth
            
        Returns:
            List[int]: Count of examples for each of the 6 fallback levels
        """
        level_counts = [0] * 6
        processed_count = 0
        error_count = 0
        
        for result_data in results_with_embeddings:
            try:
                result = result_data.get('result')
                gt_instances = result_data.get('gt_instances', [])
                
                if not result or not hasattr(result, 'pred_instances'):
                    continue
                    
                pred_instances = result.pred_instances
                
                # Extract prediction data safely
                pred_scores = pred_instances.scores.cpu().numpy()
                pred_labels = pred_instances.labels.cpu().numpy()
                pred_bboxes = pred_instances.bboxes.cpu().numpy()
                
                # Process high-confidence detections
                for bbox, score, pred_label in zip(pred_bboxes, pred_scores, pred_labels):
                    if score < min_score:
                        continue
                    
                    # Determine fallback level for this detection
                    fallback_level = self._get_fallback_level_for_detection(
                        bbox, pred_label, gt_instances, iou_threshold
                    )
                    
                    if fallback_level is not None and 0 <= fallback_level < 6:
                        level_counts[fallback_level] += 1
                        
                processed_count += 1
                        
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Only show first few errors
                    print(f"Warning: Error processing result for diversity: {e}")
                continue
        
        if error_count > 3:
            print(f"Warning: {error_count} total errors during diversity checking")
            
        return level_counts
    
    def _get_fallback_level_for_detection(self, bbox: np.ndarray, pred_label: int,
                                        gt_instances: List[Dict], iou_threshold: float) -> Optional[int]:
        """
        Get fallback level for a single detection with robust matching.
        
        Matches detection to ground truth and determines hierarchical relationship
        between predicted and actual classes.
        
        Args:
            bbox: Predicted bounding box [x1, y1, x2, y2]
            pred_label: Predicted class index
            gt_instances: Ground truth instances for the image
            iou_threshold: IoU threshold for matching
            
        Returns:
            Optional[int]: Fallback level (0-5) if valid match found, None otherwise
        """
        # Find best matching ground truth
        gt_leaf = None
        best_iou = 0.0
        
        for gt_inst in gt_instances:
            try:
                gt_bbox = gt_inst['bbox']
                iou = bbox_iou(bbox, gt_bbox)
                
                if iou > iou_threshold and iou > best_iou:
                    gt_label = gt_inst['bbox_label']
                    if 0 <= gt_label < len(self.labels):
                        gt_leaf = self.labels[gt_label]
                        best_iou = iou
                        
            except (KeyError, IndexError, TypeError) as e:
                continue
        
        if gt_leaf is None:
            return None
            
        # Validate predicted class
        if not (0 <= pred_label < len(self.labels)):
            return None
            
        pred_node = self.labels[pred_label]
        
        # Determine hierarchical relationship
        try:
            return determine_fallback_level(gt_leaf, pred_node, self.hierarchy)
        except Exception as e:
            print(f"Error determining fallback level for {gt_leaf} -> {pred_node}: {e}")
            return None


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's no intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# -----------------------------------------------------------------------------
# Fallback Level Logic (from hierarchical_prediction_distribution.py)
# -----------------------------------------------------------------------------

def determine_fallback_level(gt_leaf: str, pred_node: str, tree: HierarchyTree) -> int:
    """
    Determine the fallback level based on hierarchical relationship.
    
    This function analyzes the relationship between ground truth and predicted classes
    to classify the prediction into one of 6 hierarchical levels:
    
    Args:
        gt_leaf: Ground truth leaf class name
        pred_node: Predicted class name  
        tree: Hierarchy tree containing class relationships
        
    Returns:
        int: Fallback level classification:
            0 = Exact leaf match (perfect prediction)
            1 = Parent match (one level up)
            2 = Grandparent match (two levels up)
            3 = Sibling match (same parent)
            4 = Cousin match (same grandparent)
            5 = Off-branch (no hierarchical relationship)
    """
    # Early exit for missing classes
    if gt_leaf not in tree.class_to_node or pred_node not in tree.class_to_node:
        return 5  # off-branch
    
    # Exact match check
    if gt_leaf == pred_node:
        return 0
    
    gt_node = tree.class_to_node[gt_leaf]
    
    # Check parent relationship
    if gt_node.parent and gt_node.parent.name == pred_node:
        return 1
    
    # Check grandparent relationship
    grandparent = tree.get_grandparent(gt_leaf)
    if grandparent and grandparent == pred_node:
        return 2
    
    # Check sibling relationship (same parent)
    siblings = tree.get_siblings(gt_leaf)
    if pred_node in siblings:
        return 3
    
    # Check cousin relationship (same grandparent)
    cousins = tree.get_cousins(gt_leaf)
    if pred_node in cousins:
        return 4
    
    # No hierarchical relationship found
    return 5


def get_fallback_visual_encoding(level: int) -> Dict[str, Any]:
    """
    Get visual encoding for fallback level with color-blind friendly palette.
    
    Uses a carefully selected color scheme that maintains consistency with
    hierarchical_prediction_distribution.py while being accessible to users
    with color vision deficiencies.
    
    Args:
        level: Hierarchical fallback level (0-5)
        
    Returns:
        Dict[str, Any]: Visual encoding containing:
            - 'color': Hex color code for border/marker
            - 'linestyle': Line style for plotting
            - 'linewidth': Line width for emphasis
    """
    # Color-blind friendly palette optimized for hierarchical levels
    # Progression: Dark green (best) -> Light green -> Blue -> Purple -> Orange -> Red (worst)
    visual_encodings = {
        0: {  # Exact match - Dark green (excellent)
            'color': '#1B5E20',      # Dark green
            'linestyle': '-', 
            'linewidth': 4
        },
        1: {  # Parent match - Medium green (very good)
            'color': '#388E3C',      # Medium green  
            'linestyle': '-',
            'linewidth': 4
        },
        2: {  # Grandparent match - Light green (good)
            'color': '#66BB6A',      # Light green
            'linestyle': '-',
            'linewidth': 4
        },
        3: {  # Sibling match - Blue (acceptable)
            'color': '#1976D2',      # Blue
            'linestyle': '-',
            'linewidth': 4
        },
        4: {  # Cousin match - Purple (poor)
            'color': '#7B1FA2',      # Purple
            'linestyle': '-',
            'linewidth': 4
        },
        5: {  # Off-branch - Red (worst)
            'color': '#D32F2F',      # Red
            'linestyle': '-',
            'linewidth': 4
        }
    }
    
    # Return encoding for level, defaulting to off-branch for invalid levels
    return visual_encodings.get(level, visual_encodings[5])


def _select_balanced_subset(
    source_examples: List[Dict[str, Any]], 
    num_to_select: int
) -> List[Dict[str, Any]]:
    """
    Selects a subset of examples, attempting to maintain balance across fallback levels.
    """
    if not source_examples or num_to_select <= 0:
        return []

    # Ensure all source examples have a valid fallback_level
    valid_source_examples = [ex for ex in source_examples if 'fallback_level' in ex and 0 <= ex['fallback_level'] < 6]
    if not valid_source_examples:
        # If no valid examples, return a simple slice of the original if desperate, or empty
        return source_examples[:num_to_select] if source_examples else []

    examples_by_level = {level: [] for level in range(6)}
    for ex in valid_source_examples:
        examples_by_level[ex['fallback_level']].append(ex)

    # Sort examples within each level by confidence (descending) if available
    for level in range(6):
        if examples_by_level[level]:
            examples_by_level[level].sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

    selected_examples: List[Dict[str, Any]] = []
    # Keeps track of how many we've taken from each level's sorted list
    num_taken_from_level = [0] * 6 

    # First pass: Try to get a proportional number from each level
    # Target at least 1 per level if num_to_select allows, or num_to_select // 6
    target_per_level_first_pass = max(1, num_to_select // 6 if num_to_select >= 6 else 1)

    for level in range(6):
        if len(selected_examples) >= num_to_select:
            break 

        available_count_in_level = len(examples_by_level[level])
        
        # How many to attempt to take from this level in this pass
        num_to_attempt = min(target_per_level_first_pass, available_count_in_level - num_taken_from_level[level])
        
        # How many can we actually take given remaining overall slots
        num_can_actually_take = min(num_to_attempt, num_to_select - len(selected_examples))

        if num_can_actually_take > 0:
            start_index = num_taken_from_level[level]
            selected_examples.extend(examples_by_level[level][start_index : start_index + num_can_actually_take])
            num_taken_from_level[level] += num_can_actually_take
            
    # Second pass: Fill remaining slots by cycling through levels that still have examples
    # This loop continues as long as we need more examples and can still add them.
    while len(selected_examples) < num_to_select:
        added_in_this_cycle = False
        for level in range(6): # Cycle through levels
            if len(selected_examples) >= num_to_select:
                break # Stop if we've filled up

            # Check if this level still has examples we haven't taken
            if num_taken_from_level[level] < len(examples_by_level[level]):
                start_index = num_taken_from_level[level]
                selected_examples.append(examples_by_level[level][start_index])
                num_taken_from_level[level] += 1
                added_in_this_cycle = True
                if len(selected_examples) >= num_to_select:
                    break 
        
        if not added_in_this_cycle: 
            # If a full cycle through levels adds no new examples, we're done.
            break
            
    return selected_examples # The loops ensure we don't exceed num_to_select


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_nearest_neighbor_accuracy(query_embeddings: np.ndarray, 
                                     prototype_embeddings: np.ndarray,
                                     prediction_labels: np.ndarray, 
                                     class_labels: List[str], # ADDED: list of prototype class names
                                     space_name: str = "embedding",
                                     model: Optional[Any] = None,
                                     sample_size: int = 10) -> ValidationResult:
    """
    Validate that nearest neighbor in embedding space matches actual model predictions.
    
    This function supports both simple Euclidean distance and model-aware validation
    that includes temperature scaling and geometric bias. It returns a structured
    ValidationResult for consistent reporting.
    
    Args:
        query_embeddings: Query embeddings (N, D)  
        prototype_embeddings: Prototype embeddings (M, D)
        prediction_labels: Actual prediction labels (indices) for each query (N,)
        class_labels: List of class names corresponding to prototype_embeddings indices (M,)
        space_name: Name of the space for logging (e.g., "original", "PCA", "2D")
        model: Optional model for model-aware validation with temperature/bias
        sample_size: Number of samples to validate (default: 10)
        
    Returns:
        ValidationResult: Structured validation results with summary method
    """
    # Input validation
    if len(query_embeddings) == 0 or len(prediction_labels) == 0 or len(prototype_embeddings) == 0:
        return ValidationResult(
            space_name=space_name,
            sample_size=0,
            matches=0,
            match_rate=0.0,
            detailed_results=[]
        )
    if len(prototype_embeddings) != len(class_labels):
        print(f"Warning (validate_nearest_neighbor_accuracy): Mismatch between prototype_embeddings ({len(prototype_embeddings)}) and class_labels ({len(class_labels)}). Names might be incorrect.")
        # Proceed, but names might be unreliable or cause errors if indices are out of bounds.
    
    # Determine validation method and prepare
    use_model_aware = model is not None
    actual_sample_size = min(sample_size, len(query_embeddings), len(prediction_labels))
    matches = 0
    detailed_results = []
    
    # Get target classifier for model-aware validation
    target_classifier = None
    if use_model_aware:
        try:
            target_classifier = get_target_classifier(model)
        except Exception as e:
            print(f"Warning: Failed to get target classifier, falling back to simple validation: {e}")
            use_model_aware = False
    
    # Perform validation using unified function
    matches, detailed_results = _validate_embeddings(
        query_embeddings, prototype_embeddings, prediction_labels,
        actual_sample_size, class_labels, # Pass class_labels here
        target_classifier if use_model_aware else None
    )
    validation_type = "MODEL-AWARE" if use_model_aware and target_classifier is not None else "EUCLIDEAN"
    
    # Create result
    match_rate = matches / actual_sample_size if actual_sample_size > 0 else 0.0
    result = ValidationResult(
        space_name=f"{space_name} ({validation_type})",
        sample_size=actual_sample_size,
        matches=matches,
        match_rate=match_rate,
        detailed_results=detailed_results
    )
    
    return result


def _validate_embeddings(query_embeddings: np.ndarray,
                        prototype_embeddings: np.ndarray,
                        prediction_labels: np.ndarray, # These are the model's actual prediction (label indices) for queries
                        sample_size: int,
                        class_labels: List[str], # List of names for prototype classes
                        target_classifier: Any = None) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Unified validation function with automatic fallback.
    Tries model-aware validation first, falls back to simple Euclidean if needed.
    """
    matches = 0
    detailed_results = []
    
    # Try model-aware validation first if classifier is available
    if target_classifier is not None:
        try:
            with torch.no_grad():
                query_tensor = torch.from_numpy(query_embeddings[:sample_size]).float()
                # Ensure prototype_tensor is created correctly based on the full prototype_embeddings
                prototype_tensor = torch.from_numpy(prototype_embeddings).float() 
                
                for i in range(sample_size):
                    query_emb = query_tensor[i:i+1]
                    
                    # Calculate distances using model's method
                    distances = torch.cdist(query_emb, prototype_tensor, p=2).squeeze(0)
                    
                    # Apply temperature scaling if present
                    if (hasattr(target_classifier, 'use_temperature') and 
                        target_classifier.use_temperature and 
                        hasattr(target_classifier, 'logit_scale') and 
                        target_classifier.logit_scale is not None):
                        temp_scale = target_classifier.logit_scale.exp().clamp(max=100)
                        distances = distances * temp_scale
                    
                    # Convert to logits and find nearest neighbor
                    logits = -distances # Higher score (lower distance) is better
                    if (hasattr(target_classifier, 'geometric_bias') and 
                        target_classifier.geometric_bias is not None and
                        target_classifier.geometric_bias.shape == logits.shape): # Check shape
                        logits = logits + target_classifier.geometric_bias
                    
                    nn_idx = torch.argmax(logits).item() # Index of the NN prototype
                    actual_pred_label_idx = int(prediction_labels[i]) # Model's actual prediction for this query
                    nn_matches_pred = (nn_idx == actual_pred_label_idx)
                    
                    if nn_matches_pred:
                        matches += 1
                    
                    result_detail = {
                        'query_idx': i,
                        'prediction': actual_pred_label_idx, # Model's prediction index
                        'nearest_neighbor': nn_idx,          # NN prototype index in this space
                        'matches': nn_matches_pred,
                        'distance': distances[nn_idx].item() # Distance to the NN prototype
                    }
                    if not nn_matches_pred:
                        if 0 <= actual_pred_label_idx < len(class_labels):
                            result_detail['predicted_class_name'] = class_labels[actual_pred_label_idx]
                        else:
                            result_detail['predicted_class_name'] = f"UnknownLabelIdx({actual_pred_label_idx})"
                        if 0 <= nn_idx < len(class_labels):
                            result_detail['nn_class_name'] = class_labels[nn_idx]
                        else:
                            result_detail['nn_class_name'] = f"UnknownLabelIdx({nn_idx})"
                    detailed_results.append(result_detail)
                    
            return matches, detailed_results
            
        except Exception as e:
            print(f"Model-aware validation failed, using Euclidean fallback: {e}")
            # Fall through to Euclidean
    
    # Fallback to simple Euclidean distance validation
    detailed_results.clear() # Clear any partial results from failed model-aware attempt
    matches = 0 # Reset matches

    for i in range(sample_size):
        query = query_embeddings[i:i+1]
        distances = np.linalg.norm(prototype_embeddings - query, axis=1)
        nn_idx = np.argmin(distances) # Index of the NN prototype
        
        actual_pred_label_idx = int(prediction_labels[i]) # Model's actual prediction for this query
        nn_matches_pred = (nn_idx == actual_pred_label_idx)
        
        if nn_matches_pred:
            matches += 1
        
        result_detail = {
            'query_idx': i,
            'prediction': actual_pred_label_idx,
            'nearest_neighbor': nn_idx,
            'matches': nn_matches_pred,
            'distance': float(distances[nn_idx])
        }
        if not nn_matches_pred:
            if 0 <= actual_pred_label_idx < len(class_labels):
                result_detail['predicted_class_name'] = class_labels[actual_pred_label_idx]
            else:
                result_detail['predicted_class_name'] = f"UnknownLabelIdx({actual_pred_label_idx})"
            if 0 <= nn_idx < len(class_labels):
                result_detail['nn_class_name'] = class_labels[nn_idx]
            else:
                result_detail['nn_class_name'] = f"UnknownLabelIdx({nn_idx})"
        detailed_results.append(result_detail)
    
    return matches, detailed_results


def collect_validation_data(results_with_embeddings: List[Dict[str, Any]], 
                           config: VisualizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract query embeddings and prediction labels for validation.
    
    Args:
        results_with_embeddings: Results from run_inference_with_hooks
        config: Configuration with validation parameters
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (query_embeddings, prediction_labels)
    """
    all_query_embeddings = []
    all_prediction_labels = []
    
    for result_data in results_with_embeddings:
        result = result_data.get('result')
        if not result or not hasattr(result, 'pred_instances'):
            continue
            
        pred_instances = result.pred_instances
        
        # Check for required data
        if not hasattr(pred_instances, 'query_embeddings'):
            continue
            
        try:
            pred_scores = pred_instances.scores.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()
            query_embeddings = pred_instances.query_embeddings.numpy()
            
            # Filter by confidence score
            high_conf_mask = pred_scores >= config.min_confidence_score
            if not np.any(high_conf_mask):
                continue
                
            # Collect high-confidence data
            high_conf_embeddings = query_embeddings[high_conf_mask]
            high_conf_labels = pred_labels[high_conf_mask]
            
            all_query_embeddings.append(high_conf_embeddings)
            all_prediction_labels.append(high_conf_labels)
            
        except Exception as e:
            print(f"Warning: Error processing result data: {e}")
            continue
    
    # Combine results
    if not all_query_embeddings:
        print("Warning: No validation data collected")
        return np.array([]), np.array([])
    
    try:
        combined_embeddings = np.vstack(all_query_embeddings)
        combined_labels = np.concatenate(all_prediction_labels)
        
        print(f"Collected {len(combined_embeddings)} query embeddings "
              f"(min_score >= {config.min_confidence_score}) for validation")
        
        return combined_embeddings, combined_labels
        
    except Exception as e:
        print(f"Error combining validation data: {e}")
        return np.array([]), np.array([])


# -----------------------------------------------------------------------------
# Colour utilities (from embeddings notebook)
# -----------------------------------------------------------------------------

def build_hue_map(root: HierarchyNode, shrink: float = 0.8) -> Dict[str, float]:
    """
    Assign hues so siblings at shallow levels get strong separation.
    `shrink < 1` compresses the span passed down to each generation.
    """
    hue_map: Dict[str, float] = {}

    def _recurse(node: HierarchyNode, start: float, end: float, depth: int) -> None:
        hue_map[node.name] = (start + end) / 2
        if not node.children:
            return

        usable_span = (end - start) * (1.0 if depth == 0 else shrink)
        left_gap = ((end - start) - usable_span) / 2
        span_start = start + left_gap
        width = usable_span / len(node.children)

        sorted_children = sorted(node.children, key=lambda x: x.name)

        for idx, child in enumerate(sorted_children):
            cs = span_start + idx * width
            ce = cs + width
            _recurse(child, cs, ce, depth + 1)

    _recurse(root, 0.0, 1.0, 0)
    return hue_map


# -----------------------------------------------------------------------------
# UMAP and Prototype Embedding Functions (adapted from embeddings notebook)
# -----------------------------------------------------------------------------

def _load_model_and_initial_data(
    model_path_str: str, 
    config_path_str: str, 
    provided_prototype_embeddings: Optional[np.ndarray] = None
) -> Tuple[Optional[Any], Optional[HierarchyTree], Optional[List[str]], Optional[np.ndarray]]:
    """Loads the model, hierarchy, labels from annotations, and prototype embeddings."""
    try:
        print(f"Loading model config from: {config_path_str}")
        cfg = Config.fromfile(config_path_str)
        
        # Load hierarchy and labels using the existing helper
        hierarchy, labels_from_ann = _load_hierarchy_and_labels_from_config(cfg)
        if hierarchy is None or labels_from_ann is None:
            print("Error: Failed to load hierarchy or labels from config.")
            return None, None, None, None

        model_path_obj = pathlib.Path(model_path_str)
        if not model_path_obj.exists():
            print(f"Error: Model file not found at '{model_path_obj}'")
            return None, hierarchy, labels_from_ann, None # Return what we have so far

        print(f"Loading model from {model_path_str}")
        # Ensure config_path_str is the actual model's config for init_detector
        model = init_detector(config_path_str, model_path_str, device='cpu') # Use CPU for loading

        prototype_embeddings_to_use: Optional[np.ndarray] = None
        if provided_prototype_embeddings is not None:
            prototype_embeddings_to_use = provided_prototype_embeddings
            print(f"Using pre-computed prototype embeddings with shape {prototype_embeddings_to_use.shape}")
        else:
            if model:
                try:
                    # This logic for getting target_classifier might need to be robust
                    last_branch_idx = get_last_classification_branch_index(model)
                    target_branch_idx = last_branch_idx -1 # Assuming second-to-last
                    target_classifier = getattr(model.bbox_head.cls_branches, str(target_branch_idx))
                    prototype_embeddings_to_use = target_classifier.prototypes.detach().cpu().numpy()
                    print(f"Extracted prototype embeddings from model branch {target_branch_idx}, shape: {prototype_embeddings_to_use.shape}")
                except Exception as e:
                    print(f"Error extracting prototype embeddings from model: {e}")
                    # Continue without prototype_embeddings_to_use if extraction fails
            else:
                print("Model not loaded, cannot extract prototype embeddings.")
        
        if prototype_embeddings_to_use is not None and labels_from_ann is not None and len(prototype_embeddings_to_use) != len(labels_from_ann):
             print(f"Warning: Number of prototype embeddings ({len(prototype_embeddings_to_use)}) != number of categories from annotation ({len(labels_from_ann)})")


        return model, hierarchy, labels_from_ann, prototype_embeddings_to_use

    except Exception as e:
        print(f"Error in _load_model_and_initial_data: {e}")
        traceback.print_exc()
        return None, None, None, None


def _calculate_and_print_projection_diagnostics(
    high_dim_embeddings: np.ndarray, # Embeddings that were input to UMAP/MDS
    low_dim_projections: np.ndarray  # Output 2D projections from UMAP/MDS
):
    """Calculates and prints projection quality diagnostics."""
    if not SCIPY_AVAILABLE or len(high_dim_embeddings) < 10 or low_dim_projections is None or len(low_dim_projections) < 10:
        print("Skipping projection quality diagnostics: SciPy not available or insufficient data.")
        return

    print("\n=== Projection Quality Diagnostics ===")
    try:
        from sklearn.manifold import trustworthiness
        # Ensure n_neighbors is valid
        k_trust = min(5, len(high_dim_embeddings) - 1)
        if k_trust < 1:
             print("⚠️ Trustworthiness: Not enough samples to compute (k_trust < 1).")
        else:
            trust_score = trustworthiness(high_dim_embeddings, low_dim_projections, n_neighbors=k_trust)
            print(f"Trustworthiness (k={k_trust}): {trust_score:.4f} {'✅ Good' if trust_score > 0.87 else '⚠️ Moderate' if trust_score > 0.7 else '❌ Poor'}")
        
        from scipy.spatial.distance import pdist
        from scipy.stats import pearsonr
        
        if len(high_dim_embeddings) >= 2 and len(low_dim_projections) >= 2: # pdist needs at least 2 points
            high_d_distances = pdist(high_dim_embeddings, metric='euclidean')
            low_d_distances = pdist(low_dim_projections, metric='euclidean')

            if len(high_d_distances) > 1 and len(low_d_distances) > 1 and len(high_d_distances) == len(low_d_distances):
                distance_corr, _ = pearsonr(high_d_distances, low_d_distances)
                print(f"Distance correlation: {distance_corr:.4f} {'✅ Good' if distance_corr > 0.6 else '⚠️ Moderate' if distance_corr > 0.3 else '❌ Poor'}")
            else:
                print("⚠️ Could not compute distance correlation (insufficient or mismatched distance data).")
        else:
            print("⚠️ Distance correlation: Not enough samples to compute (need at least 2).")

        # Continuity calculation (from your existing code)
        from sklearn.neighbors import NearestNeighbors
        k_continuity = min(5, len(high_dim_embeddings) - 1)
        if k_continuity < 1:
            print("⚠️ Continuity: Not enough samples to compute (k_continuity < 1).")
        else:
            nbrs_high = NearestNeighbors(n_neighbors=k_continuity + 1, metric='euclidean').fit(high_dim_embeddings)
            nbrs_low = NearestNeighbors(n_neighbors=k_continuity + 1, metric='euclidean').fit(low_dim_projections)
            
            _, indices_high = nbrs_high.kneighbors(high_dim_embeddings)
            _, indices_low = nbrs_low.kneighbors(low_dim_projections)
            
            continuity_scores = []
            for i in range(len(high_dim_embeddings)):
                high_neighbors = set(indices_high[i][1:])
                low_neighbors = set(indices_low[i][1:])
                overlap = len(high_neighbors.intersection(low_neighbors))
                continuity_scores.append(overlap / k_continuity)
            
            continuity = np.mean(continuity_scores)
            print(f"Continuity (k={k_continuity}): {continuity:.4f} {'✅ Good' if continuity > 0.7 else '⚠️ Moderate' if continuity > 0.5 else '❌ Poor'}")

        # Overall assessment (can be simplified or kept if trust_score and distance_corr are available)
        # This part depends on trust_score and distance_corr being successfully computed.
        # For simplicity, I'll omit the direct overall assessment print here, 
        # as the individual metrics already give a good indication.
        # You can add it back if you ensure trust_score and distance_corr are defined.

    except ImportError as e:
        print(f"Some diagnostics unavailable due to missing libraries: {e}")
    except Exception as e:
        print(f"Error computing projection diagnostics: {e}")
        traceback.print_exc()
    print("=" * 40)


def _perform_dimensionality_reduction_and_validation(
    prototype_embeddings: np.ndarray,
    query_embeddings: Optional[np.ndarray],
    model: Optional[Any], # For model-aware validation
    prototype_class_labels: List[str], # Labels for prototype_embeddings
    detection_examples: Optional[List[Dict[str, Any]]], # For validation
    random_state: int,
    use_mds: bool
) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[PCA]]:
    """
    Performs dimensionality reduction (PCA + UMAP/MDS) and associated validations.
    Returns the projection model (UMAP reducer or None), combined 2D projections, and PCA transformer.
    """
    combined_embeddings_for_fitting = prototype_embeddings
    num_prototypes = len(prototype_embeddings)

    if query_embeddings is not None and len(query_embeddings) > 0:
        if query_embeddings.shape[1] != prototype_embeddings.shape[1]:
            print(f"Warning: Query embedding dim ({query_embeddings.shape[1]}) != prototype dim ({prototype_embeddings.shape[1]}). Skipping queries for fitting.")
        else:
            print(f"Including {len(query_embeddings)} query embeddings in dimensionality reduction fitting...")
            combined_embeddings_for_fitting = np.vstack([prototype_embeddings, query_embeddings])
    
    can_perform_query_validation = False
    validation_pred_labels = None
    num_valid_examples_for_val = 0 
    aligned_query_embeddings_for_validation = None 

    if query_embeddings is not None and len(query_embeddings) > 0 and detection_examples:
        _pred_labels_list = [ex['pred_label'] for ex in detection_examples if 'pred_label' in ex]
        if _pred_labels_list:
            validation_pred_labels = np.array(_pred_labels_list)
            num_valid_examples_for_val = len(validation_pred_labels)

            if num_valid_examples_for_val > 0 and len(query_embeddings) >= num_valid_examples_for_val:
                can_perform_query_validation = True
                aligned_query_embeddings_for_validation = query_embeddings[:num_valid_examples_for_val]
            else:
                # Use num_valid_examples_for_val directly as it's defined above
                print(f"⚠️  Query validation setup failed: Insufficient or mismatched query embeddings for available prediction labels. Queries: {len(query_embeddings)}, Pred Labels: {num_valid_examples_for_val}")
        else:
            print("⚠️  Query validation setup failed: No 'pred_label' found in detection_examples.")
            # num_valid_examples_for_val remains 0 from initialization
    else:
        print("⚠️  Query validation setup skipped: No query embeddings or detection examples provided.")
        # num_valid_examples_for_val remains 0 from initialization

    
    
    # === Pre-transformation Validation (Original Embedding Space) ===
    print("\n=== Pre-transformation Validation (Original Embedding Space) ===")
    if can_perform_query_validation:
        original_result = validate_nearest_neighbor_accuracy(
            aligned_query_embeddings_for_validation, prototype_embeddings, validation_pred_labels,
            prototype_class_labels, "Original", model=model, sample_size=min(10, len(aligned_query_embeddings_for_validation))
        )
        original_result.print_summary()
    else:
        print("⚠️  Skipping pre-transformation validation as conditions were not met.")

    # Apply PCA preprocessing
    preprocessed_embeddings_for_fitting = combined_embeddings_for_fitting
    pca_transformer = None
    if PCA_AVAILABLE and len(combined_embeddings_for_fitting) > 50: # Ensure enough samples for PCA
        n_components = min(50, combined_embeddings_for_fitting.shape[1], len(combined_embeddings_for_fitting) - 1)
        if n_components > 10 and combined_embeddings_for_fitting.shape[1] > n_components: # Meaningful reduction
            print(f"Applying PCA: {combined_embeddings_for_fitting.shape[1]}D -> {n_components}D on {len(combined_embeddings_for_fitting)} embeddings")
            pca_transformer = PCA(n_components=n_components, random_state=random_state)
            preprocessed_embeddings_for_fitting = pca_transformer.fit_transform(combined_embeddings_for_fitting)
            explained_var = np.sum(pca_transformer.explained_variance_ratio_)
            print(f"PCA preserved {explained_var:.1%} of variance")

            # PCA Validation (simplified here, can be expanded if needed)
            print("\n=== PCA Validation ===")
            if can_perform_query_validation: # pca_transformer is non-None if we are in this block
                preprocessed_prototypes_pca = pca_transformer.transform(prototype_embeddings)
                preprocessed_validation_queries_pca = pca_transformer.transform(aligned_query_embeddings_for_validation)
                
                pca_val_result = validate_nearest_neighbor_accuracy(
                    preprocessed_validation_queries_pca, preprocessed_prototypes_pca, validation_pred_labels,
                    prototype_class_labels, "PCA", model=None, sample_size=min(10, len(preprocessed_validation_queries_pca))
                )
                pca_val_result.print_summary()
            else:
                print("⚠️  Skipping PCA validation as conditions for query validation were not met.")
            print("=" * 40)
        else:
            print("Skipping PCA: insufficient dimensionality reduction benefit or input dim too low.")
    elif not PCA_AVAILABLE:
        print("Scikit-learn not available, skipping PCA.")
    else:
        print("Skipping PCA: insufficient samples.")

    # Dimensionality Reduction (UMAP or MDS)
    projection_model_reducer = None
    combined_2d_projections = None

    n_neighbors_val = min(15, len(preprocessed_embeddings_for_fitting) - 1 if len(preprocessed_embeddings_for_fitting) > 1 else 1)
    min_dist_val = 0.1
    if query_embeddings is not None and len(query_embeddings) > 0: # If joint fitting
        n_neighbors_val = min(10, len(preprocessed_embeddings_for_fitting) - 1 if len(preprocessed_embeddings_for_fitting) > 1 else 1)
        min_dist_val = 0.05
    
    if use_mds and PCA_AVAILABLE: # MDS typically needs PCA first if high-dim
        print(f"Using MetricMDS (on {'PCA-ed' if pca_transformer else 'original'} embeddings)...")
        mds = MDS(n_components=2, metric=True, random_state=random_state, dissimilarity='euclidean', normalized_stress='auto', n_jobs=-1)
        combined_2d_projections = mds.fit_transform(preprocessed_embeddings_for_fitting)
        print(f"MDS stress: {format(mds.stress_, '.4f') if hasattr(mds, 'stress_') else 'N/A'}")
        projection_model_reducer = None # MDS doesn't return a reusable transformer in the same way UMAP does
    else:
        print(f"Using UMAP (on {'PCA-ed' if pca_transformer else 'original'} embeddings) with n_neighbors={n_neighbors_val}, min_dist={min_dist_val}...")
        projection_model_reducer = umap.UMAP(
            random_state=random_state, n_neighbors=n_neighbors_val, min_dist=min_dist_val, metric='euclidean', spread=1.0
        )
        combined_2d_projections = projection_model_reducer.fit_transform(preprocessed_embeddings_for_fitting)
    
    if combined_2d_projections is None:
        print("Error: Dimensionality reduction failed.")
        return None, None, pca_transformer
        
    print(f"{'MDS' if use_mds and PCA_AVAILABLE else 'UMAP'} fitting completed. Output shape: {combined_2d_projections.shape}")

    print(f"\n=== Post-transformation Validation (2D Projected Space) ===")
    if can_perform_query_validation:
        
        projected_queries_2d_all = combined_2d_projections[num_prototypes:]

        if len(projected_queries_2d_all) >= num_valid_examples_for_val:
            prototypes_2d = combined_2d_projections[:num_prototypes]
            queries_2d_for_val = projected_queries_2d_all[:num_valid_examples_for_val] 

            proj_result = validate_nearest_neighbor_accuracy(
                queries_2d_for_val, prototypes_2d, validation_pred_labels,
                prototype_class_labels, "2D Projection", model=None, sample_size=min(10, len(queries_2d_for_val))
            )
            proj_result.print_summary()
        else:
            print(f"⚠️  Cannot perform 2D validation: Mismatched projected 2D queries ({len(projected_queries_2d_all)}) and prediction labels ({num_valid_examples_for_val}). Query embeddings might have been skipped during dimensionality reduction fitting.")
    else:
        print("⚠️  Skipping 2D validation as initial conditions for query validation were not met.")

    _calculate_and_print_projection_diagnostics(
        preprocessed_embeddings_for_fitting, # This is what UMAP/MDS was fit on
        combined_2d_projections
    )
        
    return projection_model_reducer, combined_2d_projections, pca_transformer


def load_embeddings_and_umap(model_path: str, config_path: str, random_state: int = 42, query_embeddings: Optional[np.ndarray] = None, prototype_embeddings: Optional[np.ndarray] = None, use_mds: bool = False, detection_examples: Optional[List] = None) -> Tuple[Optional[np.ndarray], Optional[umap.UMAP], Optional[HierarchyTree], Optional[Dict[str, float]], Optional[List[str]], Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load embeddings, fit UMAP or MDS, and prepare data structures.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to model config file (annotation file path will be extracted automatically)
        random_state: Random seed for UMAP/MDS
        query_embeddings: Optional detection query embeddings to include in projection fitting
        prototype_embeddings: Optional pre-computed prototype embeddings (if None, will load model and extract prototypes)
        use_mds: If True, use MetricMDS instead of UMAP for better distance preservation
        
    Returns:
        Tuple of (embeddings, umap_model, hierarchy, hue_map, labels, node_coords, query_projections)
        query_projections will be None if no query_embeddings were provided
        
    Note:
        Both prototype and query embeddings use the same transformation pipeline via the
        EmbeddingClassifier's 'prototypes' property, ensuring they are in the same space.
    """
    model, hierarchy, labels_from_ann, embeddings_from_load = _load_model_and_initial_data(
        model_path, config_path, provided_prototype_embeddings=prototype_embeddings
    )

    if model is None or hierarchy is None or labels_from_ann is None or embeddings_from_load is None:
        print("Error: Failed to load initial model, hierarchy, labels, or prototype embeddings. Exiting load_embeddings_and_umap.")
        return None, None, None, None, None, None, None

    # 'embeddings' from here on refers to the prototype embeddings loaded/extracted
    current_prototype_embeddings = embeddings_from_load

    # Perform dimensionality reduction and validation
    projection_model_reducer, combined_2d_projections, _ = _perform_dimensionality_reduction_and_validation(
        current_prototype_embeddings,
        query_embeddings, # Pass the original query_embeddings if available
        model,            # Pass the loaded model for model-aware validation
        labels_from_ann,  # Pass labels corresponding to current_prototype_embeddings
        detection_examples, # Pass detection examples for validation
        random_state,
        use_mds
    )
        
    if combined_2d_projections is None:
        print("Error: Dimensionality reduction pipeline failed.")
        return None, None, hierarchy, None, None, None, None # Return hierarchy as it was loaded

    # Split projections back into prototype and query parts
    num_prototypes = len(current_prototype_embeddings)
    prototype_2d_projections = combined_2d_projections[:num_prototypes]
    
    query_2d_projections_result = None
    if query_embeddings is not None and len(query_embeddings) > 0 and len(combined_2d_projections) > num_prototypes:
        query_2d_projections_result = combined_2d_projections[num_prototypes:]
        print(f"Split UMAP/MDS results: {len(prototype_2d_projections)} prototype projections, {len(query_2d_projections_result)} query projections")
        
        # Verify the query projections are reasonable
        if len(query_2d_projections_result) > 0:
            proto_x_range = (np.min(prototype_2d_projections[:, 0]), np.max(prototype_2d_projections[:, 0]))
            proto_y_range = (np.min(prototype_2d_projections[:, 1]), np.max(prototype_2d_projections[:, 1]))
            query_x_range = (np.min(query_2d_projections_result[:, 0]), np.max(query_2d_projections_result[:, 0]))
            query_y_range = (np.min(query_2d_projections_result[:, 1]), np.max(query_2d_projections_result[:, 1]))
            print(f"Prototype UMAP/MDS range: X={proto_x_range}, Y={proto_y_range}")
            print(f"Query UMAP/MDS range: X={query_x_range}, Y={query_y_range}")
    else:
        print("No query embeddings were projected or projection result length mismatch.")


    # Build hierarchy and hue map first to get all node names
    hue_map = build_hue_map(hierarchy.root)
    
    # Get ALL hierarchy node names
    all_hierarchy_nodes_names = []
    def collect_node_names_recursive(node):
        all_hierarchy_nodes_names.append(node.name)
        for child in node.children:
            collect_node_names_recursive(child)
    collect_node_names_recursive(hierarchy.root)
    
    node_name_to_umap_coords = {}
    for i, name in enumerate(labels_from_ann): # Iterate through the labels that match current_prototype_embeddings
        if i < len(prototype_2d_projections): # Ensure we don't go out of bounds
            node_name_to_umap_coords[name] = prototype_2d_projections[i]
        else:
            print(f"Warning: Mismatch between labels_from_ann and prototype_2d_projections length. Label: {name} at index {i} has no projection.")


    final_proj_labels = all_hierarchy_nodes_names # For consistency with how calculate_visual_attributes uses it.

    return current_prototype_embeddings, projection_model_reducer, hierarchy, hue_map, final_proj_labels, node_name_to_umap_coords, query_2d_projections_result


def calculate_visual_attributes(hierarchy: HierarchyTree, 
                               plotted_node_names: Set[str], 
                               node_name_to_umap_coords: Dict[str, np.ndarray],
                               all_labels: List[str], 
                               hue_map: Dict[str, float], 
                               depth_cmap) -> Tuple[List[str], np.ndarray, np.ndarray, List[Any], List[Any], int, int]:
    """Calculate marker sizes, colors, and filters data for plotting."""
    
    # Calculate depths
    depths_values = []
    for name in plotted_node_names:
        node = hierarchy.class_to_node.get(name)
        if node:
            depths_values.append(node.get_depth())
    min_depth = min(depths_values) if depths_values else 0
    max_depth = max(depths_values) if depths_values else 0

    # Calculate subtree sizes
    node_subtree_sizes = {}
    for name in plotted_node_names:
        node = hierarchy.class_to_node.get(name)
        if node:
            descendants = node.descendants()
            node_subtree_sizes[name] = len(descendants) + 1
        else:
            node_subtree_sizes[name] = 1 
    
    # Filter labels and coordinates
    filtered_labels = [name for name in all_labels if name in plotted_node_names]
    if not filtered_labels:
        return [], np.array([]), np.array([]), [], [], 0, 0
        
    filtered_coords = np.array([node_name_to_umap_coords[name] for name in filtered_labels])
    
    # Calculate marker sizes
    marker_sizes_raw = np.array([node_subtree_sizes.get(name, 1) for name in filtered_labels])
    min_marker_size = 20.0
    max_marker_size = 180.0
    
    if marker_sizes_raw.size > 0 and np.max(marker_sizes_raw) > np.min(marker_sizes_raw):
        log_s = np.log1p(marker_sizes_raw - np.min(marker_sizes_raw))
        if np.max(log_s) > 0: 
            norm_s = (log_s - np.min(log_s)) / np.max(log_s)
        else:
            norm_s = np.zeros_like(log_s)
        marker_sizes = norm_s * (max_marker_size - min_marker_size) + min_marker_size
    elif marker_sizes_raw.size > 0: 
        marker_sizes = np.full(len(marker_sizes_raw), (min_marker_size + max_marker_size) / 2)
    else: 
        marker_sizes = np.array([])

    # Calculate colors
    fill_colors = []
    edge_colors = []

    for name in filtered_labels:
        node = hierarchy.class_to_node.get(name)
        
        # Fill color based on hue
        hue_val = 0.0
        if node and node.parent and node.parent.name in hue_map:
            hue_val = hue_map[node.parent.name]
        elif node and node.name in hue_map:
            hue_val = hue_map[node.name]
        fill_colors.append(colorsys.hls_to_rgb(hue_val, 0.65, 0.85))

        # Edge color based on depth
        norm_depth = 0.5
        if node:
            current_depth = node.get_depth()
            if max_depth > min_depth:
                norm_depth = (current_depth - min_depth) / (max_depth - min_depth)
            elif max_depth == min_depth and max_depth > 0:
                norm_depth = 0.5
        edge_colors.append(depth_cmap(norm_depth))
            
    return filtered_labels, filtered_coords, marker_sizes, fill_colors, edge_colors, min_depth, max_depth


def plot_taxonomy_skeleton(ax: Axes, 
                          plotted_node_names: Set[str], 
                          hierarchy: HierarchyTree, 
                          node_name_to_umap_coords: Dict[str, np.ndarray]):
    """Overlay the taxonomy skeleton with improved styling."""
    lines_drawn = 0
    potential_lines = 0
    
    print(f"Drawing skeleton for {len(node_name_to_umap_coords)} nodes with coordinates")
    
    # Iterate through all nodes that have coordinates instead of just plotted_node_names
    for node_name in node_name_to_umap_coords.keys():
        node = hierarchy.class_to_node.get(node_name)
        if not node or not node.parent:
            continue
            
        potential_lines += 1
        
        # Check if both parent and child have coordinates
        parent_name = node.parent.name
        if parent_name in node_name_to_umap_coords:
            parent_coords = node_name_to_umap_coords[parent_name]
            child_coords = node_name_to_umap_coords[node_name]
            
            # Draw the skeleton line
            ax.plot(
                [parent_coords[0], child_coords[0]],
                [parent_coords[1], child_coords[1]],
                color='darkgrey', linewidth=1.2, alpha=0.6, zorder=1,
                solid_capstyle='round'  # Rounded line ends look better
            )
            lines_drawn += 1
                
    print(f"Found {potential_lines} potential relationships, drew {lines_drawn} skeleton lines")           
    if lines_drawn == 0 and potential_lines > 0:
        print("Warning: Found relationships but couldn't draw them. Check coordinate mapping.")


def plot_convex_hulls(ax: Axes, 
                     plotted_node_names: Set[str], 
                     hierarchy: HierarchyTree, 
                     node_name_to_umap_coords: Dict[str, np.ndarray], 
                     hue_map: Dict[str, float]):
    """Plot convex hulls for all hierarchy levels with improved styling."""
    if not SCIPY_AVAILABLE:
        return
        
    # Find all nodes that have children in the plot (at any hierarchy level)
    hull_candidates = set()
    for node_name in plotted_node_names:
        node = hierarchy.class_to_node.get(node_name)
        if not node:
            continue
            
        # Walk up the hierarchy and find all ancestors that could have hulls
        current = node
        while current.parent:
            parent = current.parent
            # Count how many children of this parent are plotted
            children_plotted = sum(1 for child in parent.children 
                                 if child.name in plotted_node_names)
            
            # Only add to hull_candidates if at least 3 children are plotted (for proper hulls)
            # or exactly 2 children (for line visualization)
            if children_plotted >= 2:
                hull_candidates.add(parent.name)
            current = parent

    print(f"Drawing convex hulls for {len(hull_candidates)} groups")

    # For each hull candidate, create a convex hull around its children
    hulls_drawn = 0
    for parent_name in hull_candidates:
        parent_node = hierarchy.class_to_node.get(parent_name)
        if not parent_node:
            continue

        # Collect coordinates of all children that are actually plotted
        child_coords = []
        child_names = []
        for child_node in parent_node.children:
            if child_node.name in node_name_to_umap_coords:
                child_coords.append(node_name_to_umap_coords[child_node.name])
                child_names.append(child_node.name)
        
        # Need at least 3 points for a convex hull, handle 2-point groups differently
        if len(child_coords) >= 3:
            points = np.array(child_coords)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                parent_hue = hue_map.get(parent_name, 0.0)
                
                # Use more attractive colors with better transparency
                hull_fill_color = colorsys.hls_to_rgb(parent_hue, 0.85, 0.9)
                hull_edge_color = colorsys.hls_to_rgb(parent_hue, max(0, 0.85 * 0.6), 0.9)

                # Plot the hull with improved styling
                ax.fill(
                    points[hull.vertices, 0], points[hull.vertices, 1],
                    facecolor=hull_fill_color, alpha=0.15, 
                    edgecolor=hull_edge_color, linewidth=1.5, zorder=0
                )
                
                # Add label inside the convex hull
                hull_center_x = np.mean(points[hull.vertices, 0])
                hull_center_y = np.mean(points[hull.vertices, 1])
                
                # Create a more readable parent name (remove underscores, capitalize)
                display_name = parent_name.replace('_', ' ').title()
                
                # Add the label with subtle styling
                ax.text(hull_center_x, hull_center_y, display_name,
                       fontsize=10, fontweight='bold', 
                       ha='center', va='center',
                       color=hull_edge_color, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.7, 
                               edgecolor=hull_edge_color, linewidth=1),
                       zorder=1)
                
                hulls_drawn += 1
                
            except Exception as e:
                print(f"Error creating convex hull for {parent_name}: {e}")
        
        elif len(child_coords) == 2:
            # Handle 2-point groups with a connecting line and label
            points = np.array(child_coords)
            parent_hue = hue_map.get(parent_name, 0.0)
            
            # Draw a line between the two points
            line_color = colorsys.hls_to_rgb(parent_hue, max(0, 0.85 * 0.6), 0.9)
            ax.plot(points[:, 0], points[:, 1], 
                   color=line_color, linewidth=2.5, alpha=0.6, zorder=0)
            
            # Add label at the midpoint
            mid_x = np.mean(points[:, 0])
            mid_y = np.mean(points[:, 1])
            display_name = parent_name.replace('_', ' ').title()
            
            ax.text(mid_x, mid_y, display_name,
                   fontsize=9, fontweight='bold', 
                   ha='center', va='center',
                   color=line_color, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='white', alpha=0.7, 
                           edgecolor=line_color, linewidth=1),
                   zorder=1)
            hulls_drawn += 1
    
    print(f"Successfully drew {hulls_drawn} convex hulls and 2-point groups")


# -----------------------------------------------------------------------------
# Detection Example Extraction and Processing
# -----------------------------------------------------------------------------

def extract_detection_examples_with_hooks(model, dataset, prototype_embeddings, 
                                        hierarchy: HierarchyTree, labels: List[str],
                                        num_examples: int = 20, min_score: float = 0.3,
                                        iou_threshold: float = 0.5):
    """
    Extract detection examples using forward hooks to get actual query embeddings.
    This is the scientifically accurate approach that replaces prototype approximations.
    """
    collector = EmbeddingCollector()
    
    # Create inference configuration
    inference_config = VisualizationConfig(
        target_examples=num_examples,
        batch_size=50,
        min_score=min_score,
        iou_threshold=iou_threshold
    )
    
    # Run inference with hooks to collect query embeddings
    results_with_embeddings = run_inference_with_hooks(
        model, dataset, collector, config=inference_config, 
        hierarchy=hierarchy, labels=labels)
    
    if not results_with_embeddings:
        return []
    
    # Check prototype embeddings characteristics
    print(f"\n=== PROTOTYPE EMBEDDINGS DEBUG ===")
    print(f"Prototype embeddings shape: {prototype_embeddings.shape}")
    print(f"Prototype mean norm: {np.mean(np.linalg.norm(prototype_embeddings, axis=1)):.4f}")
    print(f"Labels length: {len(labels)}")
    print(f"Sample labels: {labels[:5]}")  # Show first 5 labels
    
    # Check if prototype embeddings are diverse
    if len(prototype_embeddings) > 1:
        from scipy.spatial.distance import pdist
        proto_distances = pdist(prototype_embeddings, metric='euclidean')
        print(f"Prototype diversity - mean distance: {np.mean(proto_distances):.6f}")
        print(f"Prototype distance range: {np.min(proto_distances):.6f} to {np.max(proto_distances):.6f}")
    print("=" * 50)

    # Extract examples using the collected embeddings
    examples_by_level = {level: [] for level in range(6)}
    
    for result_data in results_with_embeddings:
        img_path = result_data['image_path']
        result = result_data['result']
        gt_instances = result_data['gt_instances']
        
        # Get predictions from DetDataSample
        pred_instances = result.pred_instances
        pred_bboxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()
        
        # Get actual query embeddings (this is the key improvement!)
        query_embeddings = None
        if hasattr(pred_instances, 'query_embeddings'):
            query_embeddings = pred_instances.query_embeddings.numpy()
        else:
            continue
            
        # Load image for crop extraction
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (cv2.error, Exception):
            continue

        # Process high-confidence predictions
        for i, (bbox, score, pred_label) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
            if score < min_score:
                continue
                
            # Find matching ground truth (if any)
            gt_leaf = None
            for gt_inst in gt_instances:
                gt_bbox = gt_inst['bbox']
                if bbox_iou(bbox, gt_bbox) > iou_threshold:
                    if gt_inst['bbox_label'] < len(labels):
                        gt_leaf = labels[gt_inst['bbox_label']]
                    break
            
            if gt_leaf is None:
                continue
                
            # Get predicted class name
            if pred_label < len(labels):
                pred_node = labels[pred_label]
            else:
                continue
                
            # Determine fallback level
            fallback_level = determine_fallback_level(gt_leaf, pred_node, hierarchy)
            
            # Extract crop
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img_rgb[y1:y2, x1:x2]
            crop_size = 96
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            
            # Use actual query embedding (this is the key scientific improvement!)
            if i < len(query_embeddings):
                feature_vector = query_embeddings[i]
                
                examples_by_level[fallback_level].append({
                    'crop_image': crop_resized,
                    'feature_vector': feature_vector,  # ACTUAL query embedding from model
                    'gt_leaf': gt_leaf,
                    'pred_node': pred_node,             # Class name (string)
                    'pred_label': pred_label,           # Class index (int)
                    'fallback_level': fallback_level,
                    'confidence': score,
                    'bbox': bbox,
                    'image_path': img_path
                })
    
    # Improved balanced sampling across fallback levels
    examples = []
    examples_per_level = max(1, num_examples // 6)  # Base allocation per level
    fallback_names = ['Leaf Correct', 'Parent', 'Grandparent', 'Sibling', 'Cousin', 'Off-branch']
    
    # Sort examples within each level by confidence
    for level in range(6):
        if examples_by_level[level]:
            examples_by_level[level].sort(key=lambda x: x['confidence'], reverse=True)
    
    # First pass: allocate base number per level
    level_counts = [0] * 6
    for level in range(6):
        level_examples = examples_by_level[level]
        if level_examples:
            take_count = min(examples_per_level, len(level_examples))
            examples.extend(level_examples[:take_count])
            level_counts[level] = take_count
    
    print(f"After first pass: {len(examples)}/{num_examples} examples allocated")
    
    # Second pass: fill remaining slots by cycling through levels that have more examples
    remaining_slots = num_examples - len(examples)
    rounds = 0
    max_rounds = 10  # Prevent infinite loops
    
    while remaining_slots > 0 and rounds < max_rounds:
        added_this_round = 0
        rounds += 1
        
        # Try to add one more to each level that has available examples
        for level in range(6):
            if remaining_slots <= 0:
                break
                
            level_examples = examples_by_level[level]
            if len(level_examples) > level_counts[level]:
                # Add the next best example from this level
                examples.append(level_examples[level_counts[level]])
                level_counts[level] += 1
                remaining_slots -= 1
                added_this_round += 1
                    
        # If we couldn't add any examples this round, break to avoid infinite loop
        if added_this_round == 0:
            print(f"Could not fill remaining {remaining_slots} slots - insufficient examples")
            break
    
    print(f"Final: {len(examples)} examples selected")
    
    # Show the final level breakdown
    final_counts = [0] * 6
    for ex in examples:
        final_counts[ex['fallback_level']] += 1
    
    for level in range(6):
        if final_counts[level] > 0:
            print(f"  {fallback_names[level]}: {final_counts[level]} examples")
    
    # Ensure we return exactly the requested number
    return examples[:num_examples]


def overlay_detection_thumbnails(ax: Axes, examples: List[Dict], disable_layout_adjustment: bool = False):
    """Overlay detection thumbnails with prediction labels on the UMAP plot.

    Args:
        ax: Matplotlib Axes to plot on.
        examples: List of detection example dictionaries.
        disable_layout_adjustment: If True, plots thumbnails at their exact UMAP coordinates
                                   without grid-based repositioning. Defaults to False.
    """

    # Calculate consistent thumbnail size as percentage of axis span
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axis_span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    
    # Adaptive thumbnail size based on number of examples to avoid overcrowding
    base_zoom = 0.06
    if len(examples) > 12:
        base_zoom = 0.05  # Smaller for more examples
    elif len(examples) < 8:
        base_zoom = 0.07  # Larger for fewer examples
        
    thumbnail_zoom = min(0.4, axis_span * base_zoom)
    print(f"Using thumbnail zoom {thumbnail_zoom:.3f} for {len(examples)} examples")
    
    # Get data coordinates range for better positioning
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]
    
    if not disable_layout_adjustment:
        # Group examples by approximate regions to avoid overlapping
        grid_size = int(np.ceil(np.sqrt(len(examples))))
        region_width = data_width / grid_size
        region_height = data_height / grid_size
        positions_used = {}  # To track used positions
    
    # Adjust positions to avoid overlaps - create a grid of possible positions
    positions_used = {}  # To track used positions
    
    for example in examples:
        x_orig, y_orig = example['umap_coords']
        crop = example['crop_image']
        fallback_level = example['fallback_level']
        pred_node = example['pred_node']
        gt_leaf = example['gt_leaf']

        display_x, display_y = x_orig, y_orig

        if not disable_layout_adjustment:
            # Calculate grid cell for this point
            grid_x = min(int((x_orig - xlim[0]) / region_width), grid_size-1)
            grid_y = min(int((y_orig - ylim[0]) / region_height), grid_size-1)
            grid_key = (grid_x, grid_y)
        
            # If position is already used, slightly offset
            offset_factor = 0
            original_grid_key = grid_key
            current_display_x, current_display_y = x_orig, y_orig # Start with original for adjustment
            
            while grid_key in positions_used and offset_factor <= 20: # Limit iterations
                offset_factor += 1
                angle = offset_factor * 45 
                distance_scale = 0.15 * (offset_factor // 8 + 1) # Increase distance for further attempts
                # Reduce distance if region is small to prevent jumping too far
                effective_region_dim = min(region_width, region_height) if region_width > 0 and region_height > 0 else min(data_width, data_height) * 0.1
                distance = effective_region_dim * distance_scale

                offset_x = distance * np.cos(np.radians(angle))
                offset_y = distance * np.sin(np.radians(angle))
                
                new_x_candidate = x_orig + offset_x # Offset from original true coordinate
                new_y_candidate = y_orig + offset_y
                
                # Check bounds before assigning as new grid key
                if (new_x_candidate >= xlim[0] and new_x_candidate <= xlim[1] and 
                    new_y_candidate >= ylim[0] and new_y_candidate <= ylim[1]):
                    
                    grid_x_candidate = min(int((new_x_candidate - xlim[0]) / region_width), grid_size-1)
                    grid_y_candidate = min(int((new_y_candidate - ylim[0]) / region_height), grid_size-1)
                    grid_key_candidate = (grid_x_candidate, grid_y_candidate)

                    if grid_key_candidate not in positions_used:
                        grid_key = grid_key_candidate
                        current_display_x, current_display_y = new_x_candidate, new_y_candidate
                        break 
                
                if offset_factor == 20: # Max attempts reached
                    # Fallback: place it at the center of its original grid cell with some jitter
                    # if the original cell itself is not the one causing repeated collisions
                    # This is a simplified fallback, might still overlap if original cell is crowded
                    center_x = xlim[0] + (min(int((x_orig - xlim[0]) / region_width), grid_size-1) + 0.5) * region_width
                    center_y = ylim[0] + (min(int((y_orig - ylim[0]) / region_height), grid_size-1) + 0.5) * region_height
                    jitter_x = region_width * 0.1 * np.random.uniform(-1, 1)
                    jitter_y = region_height * 0.1 * np.random.uniform(-1, 1)
                    current_display_x = center_x + jitter_x
                    current_display_y = center_y + jitter_y
                    # We don't update grid_key here, just accept the position
                    break 
            
            positions_used[grid_key] = True
            display_x, display_y = current_display_x, current_display_y
        
        
        # Get visual encoding
        encoding = get_fallback_visual_encoding(fallback_level)
        
        # Create image with border
        img_with_border = add_border_to_image(crop, encoding)
        
        # Create OffsetImage with consistent size
        imagebox = OffsetImage(img_with_border, zoom=thumbnail_zoom)
        
        # Ensure the thumbnail doesn't go off the edges of the plot
        clip_padding = 0.1  # 10% padding from edges
        x_clipped = max(xlim[0] + data_width * clip_padding, 
                       min(xlim[1] - data_width * clip_padding, display_x))
        y_clipped = max(ylim[0] + data_height * clip_padding,
                       min(ylim[1] - data_height * clip_padding, display_y))
        
        # Add the thumbnail
        ab = AnnotationBbox(imagebox, (x_clipped, y_clipped), frameon=False, zorder=10,
                           xycoords='data', pad=0.1)
        ax.add_artist(ab)
        
        # Calculate offset for text position (below the image)
        img_height_pixels = img_with_border.shape[0] * thumbnail_zoom
        bbox = ax.get_window_extent()
        fig_height_pixels = bbox.height
        
        # Better offset calculation with more space
        if fig_height_pixels > 0:
            text_offset_y = (img_height_pixels / fig_height_pixels) * data_height * 0.75
        else:
            text_offset_y = data_height * 0.03
        
        # Truncate long names for better display
        gt_leaf_display = gt_leaf[:10] + "..." if len(gt_leaf) > 12 else gt_leaf
        pred_node_display = pred_node[:10] + "..." if len(pred_node) > 12 else pred_node
        
        # Get fallback name
        fallback_names = ['Exact Match', 'Parent', 'Grandparent', 'Sibling', 'Cousin', 'Off-branch']
        fallback_name = fallback_names[fallback_level]
        
        # Create label format based on fallback level for cleaner display
        # Integrate fallback level with GT/Pred in a single box
        if fallback_level == 0:
            # For exact matches, simplified display
            label_text = f"{fallback_name}\n{gt_leaf_display}"
        else:
            # For other relationships, include both with relationship type at top
            label_text = f"{fallback_name}\nGT: {gt_leaf_display}\nPred: {pred_node_display}"
        
        # Position text below the thumbnail
        text_y = max(ylim[0] + data_height * 0.05, y_clipped - text_offset_y)
        
        # Get the color for box styling
        text_color = encoding['color']
        
        # Convert color to RGB for transparency handling
        if isinstance(text_color, str):
            rgb_color = mcolors.to_rgb(text_color)
        else:
            rgb_color = text_color
            
        # Create a lighter version for the background (same color family but lighter)
        # Make it lighter but maintain enough saturation to be distinguishable
        # Lighten the color more to improve contrast for text while keeping the hue identity
        light_color_rgb = tuple([min(1.0, c * 0.2 + 0.8) for c in rgb_color])  # 80% lighter
        
        # Single box with colored background and border matching the fallback level
        # Higher alpha value to fix transparency issue for better readability
        main_box = dict(
            boxstyle='round,pad=0.4,rounding_size=0.3',
            facecolor=light_color_rgb,  # Light version of the color for background
            edgecolor=text_color,       # Original color for border
            alpha=0.85,                 # Increased from 0.2 to 0.85 for better readability
            linewidth=2.0               # Prominent border
        )
        
        # Add the combined text box with improved styling for better readability
        text = ax.text(
            x_clipped, text_y, label_text,
            fontsize=8.5,            # Slightly larger font
            color='black',           # Black text for best contrast
            weight='bold',           # Bold for better readability 
            ha='center', 
            va='top', 
            zorder=11,
            linespacing=1.2,         # Increased line spacing for better readability
            bbox=main_box
        )
        
        # Add subtle text outline for better readability against any background
        text.set_path_effects([
            path_effects.withStroke(linewidth=0.8, foreground='white')
        ])


def add_border_to_image(img: np.ndarray, encoding: Dict[str, Any]) -> np.ndarray:
    """Add colored border to image based on fallback level encoding with publication-quality styling."""
    # Use professional border styling with improved contrast
    border_width = 6  # Slightly thinner for cleaner appearance
    color = encoding['color']
    
    # Convert color name to RGB values for OpenCV
    if isinstance(color, str):
        color_rgb = mcolors.to_rgb(color)
        color_bgr = [int(c * 255) for c in color_rgb[::-1]]  # BGR for OpenCV
    else:
        color_bgr = color
    
    # Create a crisp black outline first for better definition
    img_with_outline = cv2.copyMakeBorder(img, 1, 1, 1, 1, 
                                        cv2.BORDER_CONSTANT, value=[40, 40, 40])
    
    # Add white padding for better separation from the outline - creates a nice double-border effect
    img_with_padding = cv2.copyMakeBorder(img_with_outline, 2, 2, 2, 2, 
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Add the colored border with proper thickness
    bordered = cv2.copyMakeBorder(img_with_padding, border_width, border_width, border_width, border_width,
                                 cv2.BORDER_CONSTANT, value=color_bgr)
    
    # Add a final thin dark outline for better overall definition
    final_border = cv2.copyMakeBorder(bordered, 1, 1, 1, 1,
                                    cv2.BORDER_CONSTANT, value=[50, 50, 50])
    
    return final_border


def add_detection_border_legend(ax: Axes):
    """Add a compact legend for detection border colors with clear explanations."""
    # Import here to avoid potential circular imports
    import matplotlib.patches as patches
    
    # Legend data with improved labels that better explain the hierarchical relationships
    legend_items = [
        (0, 'Exact Match (correct prediction)', '#2E8B57'),       # SeaGreen 
        (1, 'Parent (prediction too general)', '#66CDAA'),        # MediumAquamarine
        (2, 'Grandparent (prediction very general)', '#90EE90'),  # LightGreen
        (3, 'Sibling (same parent class)', '#87CEEB'),            # SkyBlue
        (4, 'Cousin (related class)', '#DDA0DD'),                 # Plum
        (5, 'Off-branch (unrelated class)', '#DC143C')            # Crimson
    ]
    
    # Create legend elements with better styling - boxes with colored borders
    legend_elements = []
    for _level, label, color in legend_items:
        # Rectangle with white interior and colored border - matching our new display style
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                               edgecolor=color, facecolor='white', alpha=0.95)
        legend_elements.append((rect, label))
    
    # Add legend with improved styling and positioning
    legend = ax.legend([elem[0] for elem in legend_elements], 
                      [elem[1] for elem in legend_elements],
                      loc='upper left', bbox_to_anchor=(0.01, 0.99), 
                      fontsize=9, title="GT-Prediction Relationships", title_fontsize=11,
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                      ncol=1)  # Single column for better readability
    
    # Style the legend with professional polish
    legend.get_frame().set_facecolor('#fcfcfc')  # Almost white
    legend.get_frame().set_edgecolor('#888888')  # Darker gray
    legend.get_frame().set_linewidth(1.0)
    
    # Make the title bold
    title = legend.get_title()
    title.set_fontweight('bold')


def plot_fallback_arrows(
    ax: Axes,
    query_examples: List[Dict[str, Any]],
    prototype_coords: Dict[str, np.ndarray]  # Maps node names (both pred and GT) to UMAP coords
):
    """
    Draws two arrows for each query example:
    1. Predicted Prototype -> Query (solid, colored by fallback_level)
    2. Query -> Ground Truth Prototype (dashed, gray)

    Args:
        ax: Matplotlib axes to plot on.
        query_examples: List of query example dicts. Each must contain:
                        'umap_coords' (np.ndarray for query's 2D position),
                        'pred_node' (str, name of predicted prototype),
                        'gt_leaf' (str, name of ground truth leaf prototype),
                        'fallback_level' (int).
        prototype_coords: Dict mapping prototype node names to their UMAP coordinates.
    """
    arrows_drawn_pred_to_query = 0
    arrows_drawn_query_to_gt = 0

    if not query_examples:
        print("No query examples provided for fallback arrows.")
        return


    for idx, example in enumerate(query_examples): # Added enumerate for easier debugging
        query_coord = example.get('umap_coords')
        predicted_node_name = example.get('pred_node')
        gt_leaf_name = example.get('gt_leaf')
        fallback_level = example.get('fallback_level')
        image_path_short = pathlib.Path(example.get('image_path', 'N/A')).name


        if query_coord is None:
            # print(f"DEBUG Arrow Skip (Example {idx}, Img: {image_path_short}): Query UMAP coordinate is None.")
            continue
        if predicted_node_name is None:
            # print(f"DEBUG Arrow Skip (Example {idx}, Img: {image_path_short}): Predicted node name is None.")
            continue
        if gt_leaf_name is None:
            # print(f"DEBUG Arrow Skip (Example {idx}, Img: {image_path_short}): GT leaf name is None.")
            continue
        if fallback_level is None:
            # print(f"DEBUG Arrow Skip (Example {idx}, Img: {image_path_short}): Fallback level is None.")
            continue

        # --- Arrow 1: Predicted Prototype to Query ---
        predicted_prototype_coord = prototype_coords.get(predicted_node_name)
        if predicted_prototype_coord is not None:
            encoding = get_fallback_visual_encoding(fallback_level)
            arrow_color_pred = encoding['color']
            
            arrowprops_pred_to_query = dict(
                arrowstyle="->,head_length=0.7,head_width=0.4",
                color=arrow_color_pred,
                linestyle='-', 
                linewidth=2.0,
                shrinkA=1,  # Shrink from predicted prototype marker
                shrinkB=1   # Shrink to query/thumbnail center
            )
            try:
                # Arrowhead points TO query_coord, FROM predicted_prototype_coord
                ax.annotate(
                    "", 
                    xy=query_coord,                         # End point (arrowhead at query)
                    xytext=predicted_prototype_coord,       # Start point (tail at prediction)
                    arrowprops=arrowprops_pred_to_query, 
                    zorder=5.1 
                )
                arrows_drawn_pred_to_query += 1
            except Exception as e:
                print(f"Error drawing PRED_TO_QUERY arrow for {predicted_node_name} (Img: {image_path_short}): {e}")
        else:
            # print(f"DEBUG PRED_TO_QUERY Arrow (Example {idx}, Img: {image_path_short}): Not drawn. Predicted prototype '{predicted_node_name}' has no UMAP coordinate. Query at {query_coord}.")
            pass


        # --- Arrow 2: Query to Ground Truth Prototype ---
        # This arrow remains the same: from query_coord to gt_prototype_coord
        gt_prototype_coord = prototype_coords.get(gt_leaf_name)

        if gt_prototype_coord is not None:
            arrowprops_query_to_gt = dict(
                arrowstyle="->,head_length=0.6,head_width=0.3", 
                color='gray', # Or a distinct "GT" color
                linestyle='--', 
                linewidth=1.8,
                shrinkA=1, # Shrink from query/thumbnail center
                shrinkB=1  # Shrink to GT prototype marker
            )
            try:
                # Arrowhead points TO gt_prototype_coord, FROM query_coord
                ax.annotate(
                    "", 
                    xy=gt_prototype_coord, # End point (arrowhead at GT)
                    xytext=query_coord,    # Start point (tail at query)
                    arrowprops=arrowprops_query_to_gt, 
                    zorder=5 
                )
                arrows_drawn_query_to_gt += 1
            except Exception as e:
                print(f"Error drawing QUERY_TO_GT arrow for {gt_leaf_name} (Img: {image_path_short}): {e}")
        else:
            # This existing print is also helpful, let's make it more specific
            # print(f"DEBUG QUERY_TO_GT Arrow (Example {idx}, Img: {image_path_short}, GT: {gt_leaf_name}): Not drawn. GT prototype '{gt_leaf_name}' has no UMAP coordinate. Query at {query_coord}.")
            pass


def plot_prototype_scatter(ax: Axes, 
                          filtered_coords: np.ndarray, 
                          marker_sizes: np.ndarray, 
                          fill_colors: List[Any], 
                          edge_colors: List[Any], 
                          filtered_labels: List[str],
                          viz_manager: Optional['VisualizationManager'] = None):
    """
    Plot UMAP scatter points with publication-quality styling and improved readability.
    Attempts to label all nodes.
    
    Args:
        ax: Matplotlib axes to plot on
        filtered_coords: Coordinates for each prototype (N, 2)
        marker_sizes: Marker sizes for each prototype (N,)
        fill_colors: Fill colors for each prototype
        edge_colors: Edge colors for each prototype  
        filtered_labels: Label text for each prototype
        viz_manager: Optional VisualizationManager for consistent styling
    """
    if filtered_coords.size == 0 or marker_sizes.size == 0:
        print("Warning: No data to plot in plot_prototype_scatter.")
        return

    base_scale = 1.4 if viz_manager else 1.3
    adjusted_sizes = marker_sizes * base_scale
    
    ax.scatter(
        filtered_coords[:, 0], filtered_coords[:, 1],
        s=adjusted_sizes,
        c=fill_colors,
        edgecolors=edge_colors,
        linewidths=2.2,
        alpha=0.95,
        zorder=6  # Ensure prototypes are above arrows (zorder=5-5.1)
    )

    texts = []
    # Iterate through ALL filtered_labels to create text objects
    # No explicit limit on the number of labels here.
    for i in range(len(filtered_labels)):
        if i < len(filtered_coords): # Ensure coordinate exists for the label
            (x, y) = filtered_coords[i]
            lbl = filtered_labels[i]
            
            # Consistent font size for potentially many labels
            font_size = 7.5 
            
            texts.append(ax.text(
                x, y, lbl,
                fontsize=font_size, 
                color='#333333', 
                fontweight='normal',
                zorder=7 # Labels above prototype markers
            ))
    
    if texts:
        try:
            # adjust_text will try to prevent overlaps for all generated texts
            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.1, 1.1), 
                expand_text=(1.1, 1.1),
                force_points=0.15, 
                force_text=0.25,
                lim=400 # Max iterations for adjust_text
            )
        except Exception as e:
            print(f"Could not adjust text labels: {e}")


# -----------------------------------------------------------------------------
# Main Plotting Function
# -----------------------------------------------------------------------------

def _load_hierarchy_and_labels_from_config(cfg: Config) -> Tuple[Optional[HierarchyTree], Optional[List[str]]]:
    """Loads hierarchy and class labels from the annotation file specified in the config."""
    ann_file_path = None
    try:
        if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
            ann_file = cfg.test_dataloader.dataset.ann_file
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg.test_dataloader.dataset, 'data_root') else ''
            if data_root and not os.path.isabs(ann_file):
                ann_file_path = os.path.join(data_root, ann_file)
            else:
                ann_file_path = ann_file
        if ann_file_path is None:  # Fallback
            # Consider making this fallback more robust or configurable if needed
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader.dataset, 'data_root') else 'data/aircraft'
            ann_file_path = os.path.join(data_root, 'aircraft_test.json')  # Default path
            print(f"Warning: Annotation file path not found in config, using fallback: {ann_file_path}")

        if not os.path.exists(ann_file_path):
            print(f"Error: Annotation file not found at {ann_file_path}")
            return None, None

        ann_data = load(ann_file_path)
        if not isinstance(ann_data, dict) or "categories" not in ann_data or "taxonomy" not in ann_data:
            print(f"Error: Annotation file {ann_file_path} is missing 'categories' or 'taxonomy'.")
            return None, None

        categories_from_ann = ann_data["categories"]
        labels_from_ann = [cat["name"] for cat in categories_from_ann]
        hierarchy = HierarchyTree(ann_data["taxonomy"])
        print(f"Successfully loaded hierarchy and labels from {ann_file_path}")
        return hierarchy, labels_from_ann
    except Exception as e:
        print(f"Error loading hierarchy and labels from {ann_file_path or 'unknown path'}: {e}")
        return None, None


def _style_panel_axes(ax: Axes, title: str, projection_name: str):
    """Applies common styling to a panel's axes."""
    ax.set_facecolor('#f9f9f9')
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel(f"{projection_name} Dim 1", fontsize=12)
    ax.set_ylabel(f"{projection_name} Dim 2", fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#888888')
        spine.set_linewidth(0.8)


def _plot_panel1_full_structure(
    ax: Axes,
    hierarchy: HierarchyTree,
    proj_node_coords: Dict[str, np.ndarray],
    proj_labels: List[str],
    hue_map: Dict[str, float],
    depth_cmap, 
    processed_all_detection_examples: List[Dict[str, Any]],
    num_examples_display_p1p3: int,
    projection_name: str,
    model_name_stem: str
) -> Tuple[Set[str], List[str], Optional[np.ndarray], Optional[np.ndarray], List[Any], List[Any]]: # Added return type hint
    """Populates Panel 1 with the full structure visualization."""
    print("Populating Panel 1: Full Structure")

    plotted_nodes_p1 = set(proj_node_coords.keys())
    attrs_p1_list = calculate_visual_attributes(
        hierarchy, plotted_nodes_p1, proj_node_coords, proj_labels, hue_map, depth_cmap
    )
    # Ensure attrs_p1_list returns enough values or handle None if calculate_visual_attributes can return less
    if len(attrs_p1_list) == 7:
        filtered_labels_p1, filtered_coords_p1, marker_sizes_p1, \
        fill_colors_p1, edge_colors_p1, _, _ = attrs_p1_list
    else: # Fallback if calculate_visual_attributes returned unexpected number of items
        print("Warning: calculate_visual_attributes did not return expected number of items for Panel 1.")
        # Initialize to safe defaults to avoid errors, though plotting might be incomplete
        filtered_labels_p1 = []
        filtered_coords_p1 = np.array([])
        marker_sizes_p1 = np.array([])
        fill_colors_p1 = []
        edge_colors_p1 = []

    if filtered_labels_p1: 
        plot_convex_hulls(ax, plotted_nodes_p1, hierarchy, proj_node_coords, hue_map)
        plot_taxonomy_skeleton(ax, plotted_nodes_p1, hierarchy, proj_node_coords)
        plot_prototype_scatter(ax, filtered_coords_p1, marker_sizes_p1, fill_colors_p1, edge_colors_p1, filtered_labels_p1)
    else:
        ax.text(0.5, 0.5, "No prototype data for Panel 1", ha='center', va='center')

    examples_with_coords_p1 = [ex for ex in processed_all_detection_examples if 'umap_coords' in ex]
    detection_examples_for_p1_display = _select_balanced_subset(examples_with_coords_p1, num_examples_display_p1p3)
    
    if detection_examples_for_p1_display:
        print(f"Panel 1: Displaying {len(detection_examples_for_p1_display)} balanced example thumbnails.")
        p1_display_counts = [0]*6 # Assuming 6 fallback levels
        for ex_p1 in detection_examples_for_p1_display: 
            if 'fallback_level' in ex_p1 and 0 <= ex_p1['fallback_level'] < 6:
                p1_display_counts[ex_p1['fallback_level']] +=1
        print(f"  Panel 1 display distribution: {p1_display_counts}")
        overlay_detection_thumbnails(ax, detection_examples_for_p1_display, disable_layout_adjustment=False)
    
    add_detection_border_legend(ax)
    panel1_title = f"Panel 1: Full Structure ({projection_name})\n{model_name_stem}"
    _style_panel_axes(ax, panel1_title, projection_name)

    return plotted_nodes_p1, filtered_labels_p1, filtered_coords_p1, marker_sizes_p1, fill_colors_p1, edge_colors_p1


def _plot_panel2_hierarchical_retreat(
    ax: Axes,
    hierarchy: HierarchyTree,
    proj_node_coords: Dict[str, np.ndarray],
    hue_map: Dict[str, float],
    processed_all_detection_examples: List[Dict[str, Any]],
    projection_name: str,
    # Data from Panel 1 for base plot
    plotted_nodes_p1: Set[str],
    filtered_labels_p1: List[str],
    filtered_coords_p1: Optional[np.ndarray],
    marker_sizes_p1: Optional[np.ndarray],
    fill_colors_p1: List[Any],
    edge_colors_p1: List[Any]
):
    """Populates Panel 2 with the hierarchical retreat visualization."""
    print("Populating Panel 2: Parent/Ancestor Fallbacks")

    panel2_fallback_levels = {1, 2} # Focus on Parent and Grandparent
    panel2_detection_examples = [
        ex for ex in processed_all_detection_examples if ex.get('fallback_level') in panel2_fallback_levels and 'umap_coords' in ex
    ]
    print(f"Panel 2: Found {len(panel2_detection_examples)} examples (from pool of {len(processed_all_detection_examples)}) with fallback levels {panel2_fallback_levels}.")

    if filtered_labels_p1 and filtered_coords_p1 is not None and marker_sizes_p1 is not None and filtered_coords_p1.size > 0:
        plot_taxonomy_skeleton(ax, plotted_nodes_p1, hierarchy, proj_node_coords)
        plot_convex_hulls(ax, plotted_nodes_p1, hierarchy, proj_node_coords, hue_map)
        plot_prototype_scatter(ax, filtered_coords_p1, marker_sizes_p1, fill_colors_p1, edge_colors_p1, filtered_labels_p1)
    else:
        ax.text(0.5, 0.5, "No prototype data for Panel 2", ha='center', va='center')

    if panel2_detection_examples:
        print(f"Panel 2: Displaying {len(panel2_detection_examples)} example thumbnails (layout adjustment disabled) and arrows.")
        overlay_detection_thumbnails(ax, panel2_detection_examples, disable_layout_adjustment=True)
        plot_fallback_arrows(ax, panel2_detection_examples, proj_node_coords)
    else:
        print(f"No detection examples to display for Panel 2 (fallback levels {panel2_fallback_levels}).")

    add_detection_border_legend(ax) 
    panel2_title = "Panel 2: Hierarchical Retreat\n(Parent, Grandparent Fallbacks)" # Updated title slightly
    _style_panel_axes(ax, panel2_title, projection_name)


def _plot_panel3_subtree_zoom(
    ax: Axes,
    hierarchy: HierarchyTree,
    # Full projection data, will be filtered inside
    full_proj_node_coords: Dict[str, np.ndarray], 
    full_proj_labels: List[str], 
    hue_map: Dict[str, float],
    depth_cmap, # Consider adding type hint: matplotlib.colors.Colormap
    processed_all_detection_examples: List[Dict[str, Any]],
    num_examples_display_p1p3: int,
    projection_name: str,
    focus_node_name: Optional[str]
):
    """Populates Panel 3 with the focused subtree visualization."""
    print(f"Populating Panel 3: Subtree Zoom (Focus: {focus_node_name})")
    panel3_title = "" # Default title

    if focus_node_name and hierarchy.class_to_node.get(focus_node_name):
        focus_node_obj = hierarchy.class_to_node.get(focus_node_name)
        # Get descendants, ensure focus node itself is included if it's a leaf or desired
        subtree_node_names = set(focus_node_obj.descendants())
        if focus_node_name not in subtree_node_names: # Ensure focus node is part of its own "subtree"
             subtree_node_names.add(focus_node_name)

        # Filter prototype coordinates for the subtree
        sub_proj_node_coords = {
            name: coords for name, coords in full_proj_node_coords.items() 
            if name in subtree_node_names
        }
        
        # Filter detection examples whose predicted node is in the subtree
        sub_detection_examples_all_in_subtree = [
            ex for ex in processed_all_detection_examples 
            if ex.get('pred_node') in subtree_node_names and 'umap_coords' in ex
        ]
        sub_detection_examples_for_p3_display = _select_balanced_subset(
            sub_detection_examples_all_in_subtree, num_examples_display_p1p3
        )
        
        plotted_nodes_p3 = set(sub_proj_node_coords.keys())
        panel3_title = f"Panel 3: Subtree '{focus_node_name}'"
        
        if plotted_nodes_p3:
            # Pass full_proj_labels, calculate_visual_attributes will filter internally
            attrs_p3_list = calculate_visual_attributes(
                hierarchy, plotted_nodes_p3, sub_proj_node_coords, 
                full_proj_labels, hue_map, depth_cmap
            )
            
            if len(attrs_p3_list) == 7:
                filtered_labels_p3, filtered_coords_p3, marker_sizes_p3, \
                fill_colors_p3, edge_colors_p3, _, _ = attrs_p3_list
            else:
                print(f"Warning: calculate_visual_attributes returned unexpected items for Panel 3.")
                filtered_labels_p3 = [] # Safe default
                filtered_coords_p3 = np.array([])
                marker_sizes_p3 = np.array([])
                fill_colors_p3 = []
                edge_colors_p3 = []

            if filtered_labels_p3 and filtered_coords_p3.size > 0:
                plot_convex_hulls(ax, plotted_nodes_p3, hierarchy, sub_proj_node_coords, hue_map)
                plot_taxonomy_skeleton(ax, plotted_nodes_p3, hierarchy, sub_proj_node_coords)
                plot_prototype_scatter(ax, filtered_coords_p3, marker_sizes_p3, fill_colors_p3, edge_colors_p3, filtered_labels_p3)
            else:
                ax.text(0.5, 0.5, f"No prototype data for subtree '{focus_node_name}'", ha='center', va='center')

            if sub_detection_examples_for_p3_display:
                print(f"Panel 3: Displaying {len(sub_detection_examples_for_p3_display)} balanced example thumbnails for subtree.")
                p3_display_counts = [0]*6
                for ex_p3 in sub_detection_examples_for_p3_display: 
                    if 'fallback_level' in ex_p3 and 0 <= ex_p3['fallback_level'] < 6:
                         p3_display_counts[ex_p3['fallback_level']] +=1
                print(f"  Panel 3 display distribution: {p3_display_counts}")
                overlay_detection_thumbnails(ax, sub_detection_examples_for_p3_display, disable_layout_adjustment=False)
        else: # plotted_nodes_p3 is empty
            ax.text(0.5, 0.5, f"No data to display for focus node '{focus_node_name}'", ha='center', va='center', fontsize=12, color='gray')
            panel3_title = f"Panel 3: Subtree '{focus_node_name}' (Empty)"

    elif focus_node_name: # Node name given but not found in hierarchy
        ax.text(0.5, 0.5, f"Focus node '{focus_node_name}' not found in hierarchy.", ha='center', va='center', fontsize=12, color='red')
        panel3_title = "Panel 3: Invalid Focus Node"
    else: # No focus node provided
        ax.text(0.5, 0.5, "Panel 3: No focus node selected", ha='center', va='center', fontsize=12, color='gray')
        panel3_title = "Panel 3: Subtree Zoom (Select Node)"

    _style_panel_axes(ax, panel3_title, projection_name)


def generate_detection_projection_plot(
    model_path: str, 
    config_path: str,
    save_path: Optional[str] = None,
    num_examples_to_collect: int = 80, # Renamed and default increased
    num_examples_display_p1p3: int = 20, # New parameter for P1/P3 display
    random_state: int = 42,
    min_score: float = 0.3,
    iou_threshold: float = 0.5,
    use_mds: bool = False,
    focus_node_name: Optional[str] = None,
    separate_panels: bool = False
):
    """
    Create a 3-panel visualization:
    1. Full UMAP/MDS with prototype embeddings and detection examples.
    2. Placeholder for Parent/Ancestor fallback visualization.
    3. Focused UMAP/MDS view on a specific subtree.
    """
    
    print("Setting up dataset and model...")
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    model = init_detector(config_path, model_path, device='cuda') # Ensure device is appropriate

    print("Loading prototype embeddings and hierarchy for UMAP fitting...")
    # It calls load_embeddings_and_umap, extract_detection_examples_with_hooks,
    # and prepares all necessary data for the full plot.
    try:
        target_classifier_module = get_target_classifier(model) # You might need to define/refine this
        prototype_embeddings_from_model = target_classifier_module.prototypes.detach().cpu().numpy()
    except Exception as e:
        print(f"Error getting prototype_embeddings_from_model: {e}. Check model structure and get_target_classifier.")
        return

    # Load hierarchy and labels using the new helper function
    hierarchy, labels_from_ann = _load_hierarchy_and_labels_from_config(cfg)
    if hierarchy is None or labels_from_ann is None:
        print("Failed to load hierarchy or labels. Exiting.")
        return

    print("Extracting detection examples with actual query embeddings using hooks...")
    # Collect a larger pool of examples using num_examples_to_collect
    initial_all_detection_examples = extract_detection_examples_with_hooks(
        model, dataset, prototype_embeddings_from_model,
        hierarchy, labels_from_ann, 
        num_examples_to_collect, # Use the larger number here
        min_score, iou_threshold
    )

    # Filter for valid examples with feature_vectors; this becomes our main pool
    processed_all_detection_examples = []
    query_embeddings_for_umap_list = []
    if initial_all_detection_examples:
        for ex in initial_all_detection_examples:
            if 'feature_vector' in ex and ex['feature_vector'] is not None:
                processed_all_detection_examples.append(ex)
                query_embeddings_for_umap_list.append(ex['feature_vector'])
        print(f"Successfully processed {len(processed_all_detection_examples)} examples with feature vectors from {len(initial_all_detection_examples)} initial.")
    
    query_embeddings_for_umap = np.array(query_embeddings_for_umap_list) if query_embeddings_for_umap_list else None

    if not processed_all_detection_examples and not proj_node_coords: # Check if there's anything to plot
        print("No prototype or detection data to plot. Exiting.")
        # Optionally create a truly empty plot or just return
        if save_path and separate_panels: # If separate, maybe indicate no data for each panel
             base_s, ext_s = os.path.splitext(save_path)
             for i in range(1,4):
                 fig_empty, ax_empty = plt.subplots(1, 1, figsize=(12,12))
                 ax_empty.text(0.5, 0.5, f"No data for Panel {i}", ha='center', va='center')
                 empty_path = f"{base_s}_panel{i}_nodata{ext_s}"
                 fig_empty.savefig(empty_path)
                 plt.close(fig_empty)
                 print(f"Empty plot placeholder saved to {empty_path}")
        elif save_path: # Combined empty plot
             fig_empty, ax_empty = plt.subplots(1, 1, figsize=(12,12))
             ax_empty.text(0.5, 0.5, "No data to plot.", ha='center', va='center')
             empty_path = f"{save_path}_nodata.png" # Ensure different name
             fig_empty.savefig(empty_path)
             plt.close(fig_empty)
             print(f"Empty plot placeholder saved to {empty_path}")
        else:
            plt.figure(figsize=(12,12))
            plt.text(0.5,0.5, "No data to plot", ha='center', va='center')
            plt.show()
        return
    
    print(f"Performing single joint projection with {len(prototype_embeddings_from_model)} prototypes + {len(query_embeddings_for_umap) if query_embeddings_for_umap is not None else 0} query embeddings...")
    _prototype_embeds_raw, _projection_model, hierarchy_from_load, hue_map, \
    proj_labels, proj_node_coords, proj_query_coords = load_embeddings_and_umap(
        model_path, config_path, random_state, 
        query_embeddings=query_embeddings_for_umap, 
        prototype_embeddings=prototype_embeddings_from_model,
        use_mds=use_mds, 
        detection_examples=processed_all_detection_examples # Pass all processed examples for validation
    )

    if _prototype_embeds_raw is None or hierarchy_from_load is None or proj_labels is None or proj_node_coords is None or hue_map is None:
        print("Failed to load embeddings or related data structures from load_embeddings_and_umap. Exiting.")
        return
    
    hierarchy = hierarchy_from_load 

    # Assign UMAP coordinates to all processed detection examples
    if proj_query_coords is not None and len(proj_query_coords) == len(processed_all_detection_examples):
        for i, example in enumerate(processed_all_detection_examples):
            example['umap_coords'] = proj_query_coords[i]
    elif proj_query_coords is not None:
        print(f"Warning: Mismatch assigning UMAP coords. Projected queries: {len(proj_query_coords)}, Processed examples: {len(processed_all_detection_examples)}.")
        # Attempt to assign to the ones that match, others won't have 'umap_coords'
        min_len = min(len(proj_query_coords), len(processed_all_detection_examples))
        for i in range(min_len):
            processed_all_detection_examples[i]['umap_coords'] = proj_query_coords[i]


    # --- End of adapted data loading ---

    print("Calculating visual attributes for the main plot...")
    depth_cmap = plt.get_cmap('cividis_r')
    plt.style.use('seaborn-v0_8-whitegrid')

    projection_name = "MetricMDS" if use_mds else "UMAP"
    model_name_stem = pathlib.Path(model_path).stem.replace('_', ' ').title()

    # These visual attributes for Panel 1 are needed as input for Panel 2's call
    # This calculation is done once.
    _plotted_nodes_p1_base = set(proj_node_coords.keys() if proj_node_coords else {})
    _attrs_p1_list_base = calculate_visual_attributes(
        hierarchy, _plotted_nodes_p1_base, proj_node_coords if proj_node_coords else {}, proj_labels if proj_labels else [], hue_map if hue_map else {}, depth_cmap
    )
    if len(_attrs_p1_list_base) == 7:
        _filtered_labels_p1_base, _filtered_coords_p1_base, _marker_sizes_p1_base, \
        _fill_colors_p1_base, _edge_colors_p1_base, _, _ = _attrs_p1_list_base
    else:
        print("Error: Base attributes for Panel 1 could not be calculated. Plotting may be incomplete.")
        # Initialize to safe defaults to prevent crashes
        _filtered_labels_p1_base, _filtered_coords_p1_base, _marker_sizes_p1_base = [], np.array([]), np.array([])
        _fill_colors_p1_base, _edge_colors_p1_base = [], []


    panel_figsize = (14, 12) # Slightly adjusted for single panel titles and legends
    timestamp = time.strftime("%Y%m%d-%H%M%S") # For unique filenames if needed

    if separate_panels:
        # --- Plot Panel 1 Separately ---
        fig1, ax1 = plt.subplots(1, 1, figsize=panel_figsize, dpi=120)
        fig1.patch.set_facecolor('white')
        # _plot_panel1_full_structure itself calls calculate_visual_attributes.
        # We pass the necessary base data.
        _plot_panel1_full_structure(
            ax1, hierarchy, proj_node_coords if proj_node_coords else {}, proj_labels if proj_labels else [], hue_map if hue_map else {}, depth_cmap,
            processed_all_detection_examples, num_examples_display_p1p3,
            projection_name, model_name_stem
        )
        fig1.tight_layout(pad=2.0)
        if save_path:
            base, ext = os.path.splitext(save_path)
            panel1_save_path = f"{base}_panel1_{timestamp}{ext}"
            print(f"Saving Panel 1 to {panel1_save_path}")
            fig1.savefig(panel1_save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close(fig1)

        # --- Plot Panel 2 Separately ---
        fig2, ax2 = plt.subplots(1, 1, figsize=panel_figsize, dpi=120)
        fig2.patch.set_facecolor('white')
        _plot_panel2_hierarchical_retreat(
            ax2, hierarchy, proj_node_coords if proj_node_coords else {}, hue_map if hue_map else {},
            processed_all_detection_examples, projection_name,
            _plotted_nodes_p1_base, _filtered_labels_p1_base, _filtered_coords_p1_base,
            _marker_sizes_p1_base, _fill_colors_p1_base, _edge_colors_p1_base
        )
        fig2.tight_layout(pad=2.0)
        if save_path:
            base, ext = os.path.splitext(save_path)
            panel2_save_path = f"{base}_panel2_{timestamp}{ext}"
            print(f"Saving Panel 2 to {panel2_save_path}")
            fig2.savefig(panel2_save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close(fig2)

        # --- Plot Panel 3 Separately ---
        fig3, ax3 = plt.subplots(1, 1, figsize=panel_figsize, dpi=120)
        fig3.patch.set_facecolor('white')
        _plot_panel3_subtree_zoom(
            ax3, hierarchy, proj_node_coords if proj_node_coords else {}, proj_labels if proj_labels else [], hue_map if hue_map else {}, depth_cmap,
            processed_all_detection_examples, num_examples_display_p1p3,
            projection_name, focus_node_name
        )
        fig3.tight_layout(pad=2.0)
        if save_path:
            base, ext = os.path.splitext(save_path)
            panel3_save_path = f"{base}_panel3_{timestamp}{ext}"
            print(f"Saving Panel 3 to {panel3_save_path}")
            fig3.savefig(panel3_save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close(fig3)
        
        if not save_path:
             print("Separate panels generated. Interactive display for multiple separate plots is not enabled by default.")

    else: # Combined plot
        combined_figsize = (36, 12)
        fig, axes = plt.subplots(1, 3, figsize=combined_figsize, dpi=120)
        ax1, ax2, ax3 = axes.ravel()
        fig.patch.set_facecolor('white')

        # Panel 1
        # _plot_panel1_full_structure returns attributes, which are then used by _plot_panel2
        # Note: The _base attributes calculated earlier are identical to what this would return if inputs are same.
        # For consistency, we use the _base attributes.
        _plot_panel1_full_structure(
            ax1, hierarchy, proj_node_coords if proj_node_coords else {}, proj_labels if proj_labels else [], hue_map if hue_map else {}, depth_cmap,
            processed_all_detection_examples, num_examples_display_p1p3,
            projection_name, model_name_stem
        )
        
        # Panel 2
        _plot_panel2_hierarchical_retreat(
            ax2, hierarchy, proj_node_coords if proj_node_coords else {}, hue_map if hue_map else {},
            processed_all_detection_examples, projection_name,
            _plotted_nodes_p1_base, _filtered_labels_p1_base, _filtered_coords_p1_base,
            _marker_sizes_p1_base, _fill_colors_p1_base, _edge_colors_p1_base
        )
        
        # Panel 3
        _plot_panel3_subtree_zoom(
            ax3, hierarchy, proj_node_coords if proj_node_coords else {}, proj_labels if proj_labels else [], hue_map if hue_map else {}, depth_cmap,
            processed_all_detection_examples, num_examples_display_p1p3,
            projection_name, focus_node_name
        )

        fig.tight_layout(pad=3.0)
        
        if save_path:
            model_name_clean = os.path.basename(model_path).replace('.', '_').replace(' ', '_')
            base, ext = os.path.splitext(save_path) # save_path is already args.save_dir/args.save_name
            
            # Construct a descriptive name for the combined plot
            scan_count = len(processed_all_detection_examples) if processed_all_detection_examples else 0
            detailed_path = f"{base}_{model_name_clean}_3panel_combined_{num_examples_display_p1p3}disp_{scan_count}scan_{timestamp}{ext}"
            
            save_dir_actual = os.path.dirname(detailed_path)
            if save_dir_actual: os.makedirs(save_dir_actual, exist_ok=True)
            
            print(f"Saving combined 3-panel figure to {detailed_path}")
            metadata = {
                'Title': f'Combined 3-Panel {projection_name} visualization',
                'Author': 'Hierarchical Object Detection Analysis',
                'Description': f'Model: {model_name_stem}, Examples Scanned: {scan_count}, P1/P3 Display: {num_examples_display_p1p3}, Focus: {focus_node_name or "None"}, Date: {timestamp}',
                'Keywords': f'{projection_name}, embeddings, object detection, hierarchy, multi-panel'
            }
            fig.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3, metadata=metadata)
            print(f"Combined figure saved. Scanned {scan_count} examples. Displayed {num_examples_display_p1p3} in P1/P3.")
        
        plt.show()


# -----------------------------------------------------------------------------
# Command Line Interface (following hierarchical_prediction_distribution.py pattern)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Create combined UMAP visualization with detection examples using forward hooks')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('save_dir', help='Directory where the figure will be saved')
    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--save-name', default='combined_umap_visualization.png', 
                       help='Name of the saved figure (default: combined_umap_visualization.png)')
    parser.add_argument('--num-examples-display', type=int, default=20, 
                       help='Number of detection example thumbnails to display in Panel 1 and 3 (default: 20)')
    parser.add_argument('--num-examples-scan', type=int, default=80,
                       help='Total number of diverse examples to scan/collect for populating all panels (default: 80)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for UMAP (default: 42)')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum confidence score for detection examples (default: 0.3)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching detected bboxes to ground truth (default: 0.5)')
    parser.add_argument('--use-mds', action='store_true',
                       help='Use MetricMDS instead of UMAP for better distance preservation (slower but more accurate)')
    
    parser.add_argument('--focus-node', type=str, default=None,
                       help='Node name to focus on for subtree visualization (Panel 3)')
    parser.add_argument('--separate-panels', action='store_true',
                       help='Save each of the three panels as a separate image file instead of a single combined image.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, args.save_name)
    
    generate_detection_projection_plot(
        model_path=args.model_path,
        config_path=args.config,
        save_path=save_path,
        num_examples_display_p1p3=args.num_examples_display,
        num_examples_to_collect=args.num_examples_scan,
        random_state=args.random_state,
        min_score=args.min_score,
        iou_threshold=args.iou_threshold,
        use_mds=args.use_mds,
        focus_node_name=args.focus_node,
        separate_panels=args.separate_panels
    )


if __name__ == '__main__':
    main()
