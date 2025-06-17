#!/usr/bin/env python3

import argparse
import os
import pathlib
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.fileio import load
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root
from hod.utils.tree import HierarchyTree
from hod.models.layers import EmbeddingClassifier
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
    Checks diversity of fallback levels and depth levels across detection results.
    
    This class analyzes detection results to ensure adequate representation
    across all 6 hierarchical fallback levels and all relevant depth levels
    for comprehensive visualization.
    
    Attributes:
        hierarchy (HierarchyTree): Hierarchy tree for relationship analysis
        labels (List[str]): Class labels for mapping predictions
        level_names (List[str]): Human-readable names for fallback levels
        max_depth (int): Maximum depth in the hierarchy
        depth_level_names (List[str]): Human-readable names for depth levels
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
        
        # Determine max depth and depth level names
        self.max_depth = 0
        if self.hierarchy and self.hierarchy.root:
            # Calculate max depth by checking all nodes
            all_nodes = list(self.hierarchy.class_to_node.values())
            if all_nodes:
                self.max_depth = max(node.get_depth() for node in all_nodes if node) # Ensure node is not None
        self.depth_level_names = [f"Depth {d}" for d in range(self.max_depth + 1)]

        
    def check_diversity(self, results_with_embeddings: List[Dict[str, Any]], 
                       min_score: float, iou_threshold: float, 
                       min_per_level: int = 2) -> bool:
        """
        Check if we have sufficient diversity across fallback levels AND depth levels.
        
        Args:
            results_with_embeddings: Results from inference with embeddings
            min_score: Minimum confidence score for including detections
            iou_threshold: IoU threshold for matching predictions to ground truth
            min_per_level: Minimum examples required per fallback level AND per depth level
            
        Returns:
            bool: True if all levels (fallback and depth) have sufficient examples
        """
        try:
            # --- Fallback Level Diversity ---
            fallback_level_counts = self._count_fallback_levels(
                results_with_embeddings, min_score, iou_threshold
            )
            sufficient_fallback_levels = sum(1 for count in fallback_level_counts if count >= min_per_level)
            all_fallback_covered = sufficient_fallback_levels == len(self.level_names)
            
            print(f"\n--- Fallback Level Diversity ({min_per_level}+ examples per level) ---")
            print(f"Status: {sufficient_fallback_levels}/{len(self.level_names)} fallback levels have sufficient examples.")
            for name, count in zip(self.level_names, fallback_level_counts):
                status_char = "✓" if count >= min_per_level else "✗"
                print(f"  {status_char} {name}: {count} examples")

            # --- Depth Level Diversity ---
            depth_level_counts = self._count_depth_levels(
                results_with_embeddings, min_score
            )
            sufficient_depth_levels = sum(1 for count in depth_level_counts if count >= min_per_level)
            all_depths_covered = sufficient_depth_levels == (self.max_depth + 1)

            print(f"\n--- Depth Level Diversity ({min_per_level}+ examples per level) ---")
            print(f"Status: {sufficient_depth_levels}/{self.max_depth + 1} depth levels have sufficient examples.")
            for i, count in enumerate(depth_level_counts):
                status_char = "✓" if count >= min_per_level else "✗"
                print(f"  {status_char} Depth {i}: {count} examples")
            
            overall_diversity_met = all_fallback_covered and all_depths_covered
            if overall_diversity_met:
                print("\n✓ Overall diversity criteria met for both fallback and depth levels.")
            else:
                print("\n✗ Overall diversity criteria NOT YET MET.")
                
            return overall_diversity_met
            
        except Exception as e:
            print(f"Error checking diversity: {e}")
            traceback.print_exc()
            return False

    def _count_depth_levels(self, results_with_embeddings: List[Dict[str, Any]],
                             min_score: float) -> List[int]:
        """
        Count examples for each depth level of the predicted class.
        """
        depth_counts = [0] * (self.max_depth + 1)
        error_count = 0

        for result_data in results_with_embeddings:
            try:
                result = result_data.get('result')
                if not result or not hasattr(result, 'pred_instances'):
                    continue
                
                pred_instances = result.pred_instances
                pred_scores = pred_instances.scores.cpu().numpy()
                pred_labels_idx = pred_instances.labels.cpu().numpy()

                for score, pred_label_idx in zip(pred_scores, pred_labels_idx):
                    if score < min_score:
                        continue
                    
                    if not (0 <= pred_label_idx < len(self.labels)):
                        continue # Invalid label index
                    
                    pred_label_name = self.labels[pred_label_idx]
                    
                    if pred_label_name in self.hierarchy.class_to_node:
                        node = self.hierarchy.class_to_node[pred_label_name]
                        if node: # Ensure node is not None
                            depth = node.get_depth()
                            if 0 <= depth <= self.max_depth:
                                depth_counts[depth] += 1
            except Exception as e:
                error_count +=1
                if error_count <=3:
                    print(f"Warning: Error processing result for depth diversity: {e}")
                continue
        
        if error_count > 3:
            print(f"Warning: {error_count} total errors during depth diversity counting.")
        return depth_counts
    
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
    num_to_select: int,
    hierarchy: Optional[HierarchyTree] = None # Pass hierarchy to get max_depth if needed
) -> List[Dict[str, Any]]:
    """
    Selects a subset of examples, attempting to maintain balance across
    a combination of fallback levels and depth levels.
    """
    if not source_examples or num_to_select <= 0:
        return []

    # Ensure all source examples have valid fallback_level and depth_level
    # 'depth_level' should be added to examples before calling this function.
    valid_source_examples = [
        ex for ex in source_examples 
        if 'fallback_level' in ex and 0 <= ex['fallback_level'] < 6 and 'depth_level' in ex
    ]
    if not valid_source_examples:
        return source_examples[:num_to_select] if source_examples else []

    # Determine the range of depth levels present or use max_depth from hierarchy
    present_depths = sorted(list(set(ex['depth_level'] for ex in valid_source_examples)))
    if not present_depths: # Should not happen if valid_source_examples is not empty
        max_depth_in_data = 0
    else:
        max_depth_in_data = present_depths[-1]
    
    # If hierarchy is provided, we can use its max_depth for a more stable category definition
    # For now, we'll work with depths present in the data.
    # A more robust version might take max_depth as an argument.

    # Create combined categories: (fallback_level, depth_level)
    examples_by_category = {}
    for ex in valid_source_examples:
        category_key = (ex['fallback_level'], ex['depth_level'])
        if category_key not in examples_by_category:
            examples_by_category[category_key] = []
        examples_by_category[category_key].append(ex)

    # Sort examples within each category by confidence (descending)
    for category_key in examples_by_category:
        examples_by_category[category_key].sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

    selected_examples: List[Dict[str, Any]] = []
    num_taken_from_category = {key: 0 for key in examples_by_category.keys()}
    
    # Get a sorted list of unique categories to iterate over consistently
    unique_categories = sorted(list(examples_by_category.keys()))
    if not unique_categories:
        return [] # No categories to select from

    # First pass: Try to get a proportional number from each category
    num_categories = len(unique_categories)
    target_per_category_first_pass = max(1, num_to_select // num_categories if num_to_select >= num_categories else 1)

    for category_key in unique_categories:
        if len(selected_examples) >= num_to_select:
            break 

        available_count_in_category = len(examples_by_category[category_key])
        
        num_to_attempt = min(target_per_category_first_pass, available_count_in_category - num_taken_from_category[category_key])
        num_can_actually_take = min(num_to_attempt, num_to_select - len(selected_examples))

        if num_can_actually_take > 0:
            start_index = num_taken_from_category[category_key]
            selected_examples.extend(examples_by_category[category_key][start_index : start_index + num_can_actually_take])
            num_taken_from_category[category_key] += num_can_actually_take
            
    # Second pass: Fill remaining slots by cycling through categories
    while len(selected_examples) < num_to_select:
        added_in_this_cycle = False
        for category_key in unique_categories:
            if len(selected_examples) >= num_to_select:
                break

            if num_taken_from_category[category_key] < len(examples_by_category[category_key]):
                start_index = num_taken_from_category[category_key]
                selected_examples.append(examples_by_category[category_key][start_index])
                num_taken_from_category[category_key] += 1
                added_in_this_cycle = True
                if len(selected_examples) >= num_to_select:
                    break 
        
        if not added_in_this_cycle: 
            break
            
    return selected_examples

# -----------------------------------------------------------------------------
# UMAP and Prototype Embedding Functions (adapted from embeddings notebook)
# -----------------------------------------------------------------------------

def _load_model_and_initial_data(
    model_path_str: str, 
    config_path_str: str, 
    provided_prototype_embeddings: Optional[np.ndarray] = None
) -> Tuple[Optional[Config], Optional[Any], Optional[HierarchyTree], Optional[List[str]], Optional[np.ndarray]]:
    """Loads the config, model, hierarchy, labels from annotations, and prototype embeddings."""
    cfg = None
    model = None
    hierarchy = None
    labels_from_ann = None
    prototype_embeddings_to_use = None
    try:
        print(f"Loading model config from: {config_path_str}")
        cfg = Config.fromfile(config_path_str)
        cfg = replace_cfg_vals(cfg)
        update_data_root(cfg)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        
        hierarchy, labels_from_ann = _load_hierarchy_and_labels_from_config(cfg)
        if hierarchy is None or labels_from_ann is None:
            print("Error: Failed to load hierarchy or labels from config.")
            # Return cfg even if other parts fail, it might be useful for debugging.
            return cfg, None, None, None, None

        model_path_obj = pathlib.Path(model_path_str)
        if not model_path_obj.exists():
            print(f"Error: Model file not found at '{model_path_obj}'")
            return cfg, None, hierarchy, labels_from_ann, None

        print(f"Loading model from {model_path_str}")
        model = init_detector(config_path_str, model_path_str, device='cpu') # Load on CPU

        if provided_prototype_embeddings is not None:
            prototype_embeddings_to_use = provided_prototype_embeddings
            print(f"Using pre-computed prototype embeddings with shape {prototype_embeddings_to_use.shape}")
        elif model:
            target_classifier = get_target_classifier(model)
            if target_classifier and hasattr(target_classifier, 'prototypes'):
                prototype_embeddings_to_use = target_classifier.prototypes.detach().cpu().numpy()
                print(f"Extracted prototype embeddings via get_target_classifier, shape: {prototype_embeddings_to_use.shape}")
            else:
                print("Warning: Could not get target classifier or its prototypes from the model via get_target_classifier.")
        else:
            print("Model not loaded and no pre-computed prototypes provided, cannot get prototype embeddings.")
        
        if prototype_embeddings_to_use is not None and labels_from_ann is not None and len(prototype_embeddings_to_use) != len(labels_from_ann):
             print(f"Warning: Number of prototype embeddings ({len(prototype_embeddings_to_use)}) != number of categories from annotation ({len(labels_from_ann)})")

        return cfg, model, hierarchy, labels_from_ann, prototype_embeddings_to_use

    except Exception as e:
        print(f"Error in _load_model_and_initial_data: {e}")
        traceback.print_exc()
        return cfg, model, hierarchy, labels_from_ann, prototype_embeddings_to_use # Return what's available

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


def _prepare_norm_vs_depth_data(
    model_path_str: str, 
    config_path_str: str
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[HierarchyTree], Optional[List[str]], Optional[Config], Optional[Any]]:
    """Prepares data for the norm vs. depth plot."""
    print("Preparing data for norm vs. depth plot...")
    cfg, model, hierarchy, labels_from_ann, prototype_embeddings = _load_model_and_initial_data(
        model_path_str=model_path_str,
        config_path_str=config_path_str
    )

    if not all([cfg, model, hierarchy, labels_from_ann, prototype_embeddings is not None]):
        print("Error: Failed to load necessary data/model for norm_vs_depth data preparation.")
        return None, None, None, None, None, None

    if len(prototype_embeddings) != len(labels_from_ann):
        print(f"Warning: Mismatch between number of prototype embeddings ({len(prototype_embeddings)}) "
              f"and number of labels ({len(labels_from_ann)}). Attempting to proceed with matched pairs for norm plot.")

    depths = []
    norms = []
    num_labels_to_process = min(len(prototype_embeddings), len(labels_from_ann))

    for i in range(num_labels_to_process):
        label_name = labels_from_ann[i]
        embedding = prototype_embeddings[i]

        if label_name in hierarchy.class_to_node:
            node = hierarchy.class_to_node[label_name]
            if node: # Ensure node is not None
                node_depth = node.get_depth()
                embedding_norm = np.linalg.norm(embedding)
                depths.append(node_depth)
                norms.append(embedding_norm)
            # else: # This case should ideally not happen if label_name is in class_to_node
            #     print(f"Warning: Node for '{label_name}' is None in hierarchy. Skipping for norm plot.")
        else:
            print(f"Warning: Label '{label_name}' from model not found in hierarchy. Skipping for norm plot.")
    
    if not depths:
        print("No valid data points (depths/norms) prepared.")
        return None, None, hierarchy, labels_from_ann, cfg, model # Return what we have
        
    print(f"Successfully prepared {len(depths)} data points for norm vs. depth plot.")
    return depths, norms, hierarchy, labels_from_ann, cfg, model

def _plot_norm_vs_depth_on_ax(ax: plt.Axes, depths: List[float], norms: List[float], title_suffix: str = ""):
    """Plots norm vs. depth data on a given Matplotlib Axes object."""
    ax.scatter(depths, norms, alpha=0.7, edgecolors='w', s=60, c='teal')
    ax.set_xlabel("Depth of Class in Hierarchy", fontsize=12)
    ax.set_ylabel("L2 Norm of Prototype Embedding", fontsize=12)
    title = "Prototype Norm vs. Depth"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

def _prepare_margin_vs_depth_data(
    model_path_str: str,
    config_path_str: str,
    num_examples_to_collect: int,
    num_examples_to_process: int,
    min_score_threshold: float, # This is for filtering detections before margin calculation
    iou_threshold_for_collection: float, # This is for fallback level determination & diversity check
    batch_size_for_collection: int,
    max_batches_for_collection: int,
    random_state_for_collection: int
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[Config], Optional[Any], Optional[HierarchyTree], Optional[List[str]]]:
    """Prepares data for the margin vs. depth plot."""
    print("Preparing data for margin vs. depth plot...")
    loaded_cfg, model, hierarchy, labels_from_ann, _ = _load_model_and_initial_data(
        model_path_str=model_path_str,
        config_path_str=config_path_str
    )

    if not all([loaded_cfg, model, hierarchy, labels_from_ann]):
        print("Error: Failed to load necessary data/model for margin_vs_depth data preparation.")
        return None, None, None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = DATASETS.build(loaded_cfg.test_dataloader.dataset)
    collector = EmbeddingCollector()
    
    # Note: The min_score in VisualizationConfig is used by DiversityChecker.
    # The min_score_threshold argument of this function is used later for filtering detections.
    vis_config_for_collection = VisualizationConfig(
        target_examples=num_examples_to_collect,
        batch_size=batch_size_for_collection,
        max_batches=max_batches_for_collection,
        min_score=min_score_threshold, # Pass the CLI min_score here for DiversityChecker
        iou_threshold=iou_threshold_for_collection,
        diversity_check_threshold=max(10, num_examples_to_collect // 4),
        random_state=random_state_for_collection
    )

    print("Collecting detection data using run_inference_with_hooks for margin plot...")
    all_image_results = run_inference_with_hooks(
        model, dataset, collector,
        config=vis_config_for_collection,
        hierarchy=hierarchy,
        labels=labels_from_ann
    )

    if not all_image_results:
        print("No detection data collected for margin plot.")
        return None, None, loaded_cfg, model, hierarchy, labels_from_ann
    
    print(f"Collected data for {len(all_image_results)} images for margin plot.")

    all_valid_detections_for_selection = []
    print(f"Preprocessing collected detections for margin plot (filtering by score >= {min_score_threshold})...")
    temp_detection_id = 0

    for image_idx, image_data in enumerate(all_image_results):
        result = image_data.get('result')
        gt_instances_for_image = image_data.get('gt_instances', [])
        if not result or not hasattr(result, 'pred_instances'): continue
        pred_instances = result.pred_instances
        if not hasattr(pred_instances, 'query_embeddings') or pred_instances.query_embeddings is None: continue
        
        instance_scores = pred_instances.scores.cpu().numpy()
        instance_labels_idx = pred_instances.labels.cpu().numpy()
        instance_query_embeddings = pred_instances.query_embeddings.cpu()
        instance_bboxes = pred_instances.bboxes.cpu().numpy()

        for i in range(len(instance_scores)):
            score = instance_scores[i]
            if score < min_score_threshold: # Filtering based on the function argument
                continue
            pred_label_idx = instance_labels_idx[i]
            query_embedding = instance_query_embeddings[i]
            bbox = instance_bboxes[i]

            if not (0 <= pred_label_idx < len(labels_from_ann)): continue
            pred_label_name = labels_from_ann[pred_label_idx]

            current_fallback_level = 5 
            best_iou_for_fallback = 0.0
            matched_gt_leaf_for_fallback = None
            for gt_inst in gt_instances_for_image:
                try:
                    gt_bbox_val = gt_inst['bbox'] # Renamed to avoid conflict
                    iou_val = bbox_iou(bbox, gt_bbox_val) # Renamed to avoid conflict
                    if iou_val > iou_threshold_for_collection and iou_val > best_iou_for_fallback:
                        gt_label_idx = gt_inst['bbox_label']
                        if 0 <= gt_label_idx < len(labels_from_ann):
                            matched_gt_leaf_for_fallback = labels_from_ann[gt_label_idx]
                            best_iou_for_fallback = iou_val
                except Exception: continue
            if matched_gt_leaf_for_fallback:
                current_fallback_level = determine_fallback_level(matched_gt_leaf_for_fallback, pred_label_name, hierarchy)

            current_depth_level = -1
            if pred_label_name in hierarchy.class_to_node:
                node = hierarchy.class_to_node[pred_label_name]
                if node: current_depth_level = node.get_depth()
            if current_depth_level == -1: continue

            all_valid_detections_for_selection.append({
                'query_embedding': query_embedding, 'pred_label_name': pred_label_name,
                'pred_label_idx': pred_label_idx, 'score': score, 'confidence': score,
                'fallback_level': current_fallback_level, 'depth_level': current_depth_level,
                'id': temp_detection_id
            })
            temp_detection_id += 1
    
    print(f"Preprocessed {len(all_valid_detections_for_selection)} valid detections for potential selection for margin plot.")
    if not all_valid_detections_for_selection:
        print("No valid detections after filtering to process for margin.")
        return None, None, loaded_cfg, model, hierarchy, labels_from_ann

    examples_to_process_for_margin = _select_balanced_subset(
        all_valid_detections_for_selection, num_examples_to_process, hierarchy
    ) if len(all_valid_detections_for_selection) > num_examples_to_process else all_valid_detections_for_selection
    
    if not examples_to_process_for_margin:
        print("No examples selected for margin processing after balancing/selection.")
        return None, None, loaded_cfg, model, hierarchy, labels_from_ann
    print(f"Selected {len(examples_to_process_for_margin)} examples for margin calculation.")

    margins = []
    margin_depths = []
    target_cls_module = get_target_classifier(model)
    if not isinstance(target_cls_module, EmbeddingClassifier) or \
       not hasattr(target_cls_module, 'get_distance_logits') or \
       not hasattr(target_cls_module, 'prototypes'):
        print(f"Error: Target classifier module for margin calculation is not valid.")
        return None, None, loaded_cfg, model, hierarchy, labels_from_ann

    print(f"Calculating margins for {len(examples_to_process_for_margin)} selected examples...")
    for example_data in examples_to_process_for_margin:
        query_embedding_tensor = example_data['query_embedding']
        pred_label_name = example_data['pred_label_name'] # For debugging
        try:
            current_query_tensor_for_logits = query_embedding_tensor.to(device)
            if current_query_tensor_for_logits.ndim == 1: current_query_tensor_for_logits = current_query_tensor_for_logits.unsqueeze(0).unsqueeze(0)
            elif current_query_tensor_for_logits.ndim == 2 and current_query_tensor_for_logits.shape[0] == 1: current_query_tensor_for_logits = current_query_tensor_for_logits.unsqueeze(1)

            with torch.no_grad():
                prototypes_for_logits = target_cls_module.prototypes.to(device)
                logits = target_cls_module.get_distance_logits(current_query_tensor_for_logits, prototypes_for_logits)
                logits = logits.squeeze() # Squeeze out batch and query_pos dimensions
            
            probabilities = torch.softmax(logits, dim=-1)
            if probabilities.numel() < 2: continue # Need at least two probabilities for margin
            sorted_probs, _ = torch.sort(probabilities, descending=True)
            margin = (sorted_probs[0] - sorted_probs[1]).item()
            node_depth = example_data['depth_level'] 
            margins.append(margin)
            margin_depths.append(node_depth)
        except Exception as e:
            print(f"Error processing detection for margin (label: {pred_label_name}): {e}")
            # traceback.print_exc() # Can be noisy, enable if needed
            continue

    if not margins or not margin_depths:
        print("No valid data points (margins/depths) prepared for margin plot.")
        return None, None, loaded_cfg, model, hierarchy, labels_from_ann
        
    print(f"Successfully prepared {len(margins)} data points for margin vs. depth plot.")
    return margin_depths, margins, loaded_cfg, model, hierarchy, labels_from_ann

def _plot_margin_vs_depth_on_ax(ax: plt.Axes, depths: List[float], margins: List[float], title_suffix: str = ""):
    """Plots margin vs. depth data on a given Matplotlib Axes object."""
    ax.scatter(depths, margins, alpha=0.7, edgecolors='w', s=60, c='royalblue')
    ax.set_xlabel("Depth of Predicted Class in Hierarchy", fontsize=12)
    ax.set_ylabel("Prediction Margin (Top1 - Top2 Prob)", fontsize=12)
    title = "Prediction Margin vs. Depth"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

# --- Updated original plotting functions to use helpers ---
def generate_norm_vs_depth_plot(model_path: str, config_path: str, save_path: Optional[str] = None):
    """Generates and saves/shows a scatter plot of prototype L2 norm vs. hierarchical depth."""
    print(f"Generating stand-alone norm vs. depth plot...")
    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    try:
        depths, norms, _, _, _, _ = _prepare_norm_vs_depth_data(model_path, config_path)

        if depths is None or norms is None or not depths:
            print("Error: No data points to plot for norm_vs_depth. Exiting.")
            return

        print(f"Plotting {len(depths)} norm vs. depth data points.")
        fig, ax = plt.subplots(figsize=(10, 7))
        _plot_norm_vs_depth_on_ax(ax, depths, norms)
        plt.tight_layout()

        if save_path:
            print(f"Saving norm vs. depth plot to {save_path}")
            fig.savefig(save_path, dpi=300)
            print("Plot saved.")
        else:
            print("Displaying norm vs. depth plot.")
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"An error occurred in generate_norm_vs_depth_plot: {e}")
        traceback.print_exc()


def generate_margin_vs_depth_plot(
    model_path: str, config_path: str, save_path: Optional[str] = None,
    num_examples_to_collect: int = 80, num_examples_to_process: int = 20,
    min_score_threshold: float = 0.3, iou_threshold_for_collection: float = 0.5,
    batch_size_for_collection: int = 10, max_batches_for_collection: int = 30,
    random_state_for_collection: int = 42
):
    """Generates and saves/shows a scatter plot of prediction margin vs. hierarchical depth."""
    print(f"Generating stand-alone margin vs. depth plot...")
    # ... (print statements for args) ...
    try:
        margin_depths, margins, _, _, _, _ = _prepare_margin_vs_depth_data(
            model_path, config_path, num_examples_to_collect, num_examples_to_process,
            min_score_threshold, iou_threshold_for_collection, batch_size_for_collection,
            max_batches_for_collection, random_state_for_collection
        )
        if margin_depths is None or margins is None or not margin_depths:
            print("Error: No data points to plot for margin_vs_depth. Exiting.")
            return

        print(f"Plotting {len(margins)} margin vs. depth data points.")
        fig, ax = plt.subplots(figsize=(10, 7))
        _plot_margin_vs_depth_on_ax(ax, margin_depths, margins)
        plt.tight_layout()

        if save_path:
            print(f"Saving margin vs. depth plot to {save_path}")
            fig.savefig(save_path, dpi=300)
            print("Plot saved.")
        else:
            print("Displaying margin vs. depth plot.")
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"An error occurred in generate_margin_vs_depth_plot: {e}")
        traceback.print_exc()

# --- New Combined Plot Function ---
def generate_combined_depth_analysis_plot(
    model_path: str, config_path: str, save_dir: str, base_save_name: Optional[str] = None,
    separate_panels: bool = False,
    num_examples_to_collect: int = 80, num_examples_to_process: int = 20,
    min_score_threshold: float = 0.3, iou_threshold_for_collection: float = 0.5,
    batch_size_for_collection: int = 10, max_batches_for_collection: int = 30,
    random_state_for_collection: int = 42
):
    """Generates norm vs. depth and margin vs. depth plots, either combined or separately."""
    print("--- Generating Combined Depth Analysis ---")
    print(f"Model: {model_path}, Config: {config_path}, Save Dir: {save_dir}")
    print(f"Separate Panels: {separate_panels}")

    # 1. Prepare norm vs. depth data
    norm_depths, norms, _, _, _, _ = _prepare_norm_vs_depth_data(model_path, config_path)
    
    # 2. Prepare margin vs. depth data
    margin_depths, margins, _, _, _, _ = _prepare_margin_vs_depth_data(
        model_path, config_path, num_examples_to_collect, num_examples_to_process,
        min_score_threshold, iou_threshold_for_collection, batch_size_for_collection,
        max_batches_for_collection, random_state_for_collection
    )

    has_norm_data = norm_depths is not None and norms is not None and len(norm_depths) > 0
    has_margin_data = margin_depths is not None and margins is not None and len(margin_depths) > 0

    if not has_norm_data and not has_margin_data:
        print("No data available for any plot. Exiting combined plot generation.")
        return

    # Determine base name for saving
    name_part = "depth_analysis" # Default base
    if base_save_name:
        name_part = os.path.splitext(base_save_name)[0]
    
    if separate_panels:
        print("Generating plots as separate panels...")
        if has_norm_data:
            fig_norm, ax_norm = plt.subplots(figsize=(10, 7))
            _plot_norm_vs_depth_on_ax(ax_norm, norm_depths, norms, title_suffix="Separate")
            plt.tight_layout()
            norm_save_path = os.path.join(save_dir, f"{name_part}_norm_vs_depth.png")
            print(f"Saving norm vs. depth plot to {norm_save_path}")
            fig_norm.savefig(norm_save_path, dpi=300)
            plt.close(fig_norm)
        else:
            print("Skipping norm vs. depth plot (no data).")

        if has_margin_data:
            fig_margin, ax_margin = plt.subplots(figsize=(10, 7))
            _plot_margin_vs_depth_on_ax(ax_margin, margin_depths, margins, title_suffix="Separate")
            plt.tight_layout()
            margin_save_path = os.path.join(save_dir, f"{name_part}_margin_vs_depth.png")
            print(f"Saving margin vs. depth plot to {margin_save_path}")
            fig_margin.savefig(margin_save_path, dpi=300)
            plt.close(fig_margin)
        else:
            print("Skipping margin vs. depth plot (no data).")
    else:
        print("Generating plots as subplots in a single figure...")
        num_subplots = 0
        if has_norm_data: num_subplots += 1
        if has_margin_data: num_subplots += 1

        if num_subplots == 0: # Should be caught by earlier check, but as a safeguard
            print("No data to plot in combined panel.")
            return
        
        fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 7), squeeze=False) # Ensure axes is always 2D-like
        
        current_ax_idx = 0
        if has_norm_data:
            _plot_norm_vs_depth_on_ax(axes[0, current_ax_idx], norm_depths, norms)
            current_ax_idx +=1
        
        if has_margin_data:
            _plot_margin_vs_depth_on_ax(axes[0, current_ax_idx], margin_depths, margins)
            # current_ax_idx +=1 # Not needed if it's the last one
            
        fig.suptitle("Hierarchical Depth Analysis", fontsize=16, y=0.98) # y adjusted for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
        
        combined_save_path = os.path.join(save_dir, f"{name_part}_subplots.png")
        print(f"Saving combined plot to {combined_save_path}")
        fig.savefig(combined_save_path, dpi=300)
        plt.close(fig)
    
    print("Combined depth analysis plot generation finished.")


# -----------------------------------------------------------------------------
# Command Line Interface (following hierarchical_prediction_distribution.py pattern)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth analysis plots (norm vs. depth and margin vs. depth) for hierarchical object detection models.')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('save_dir', help='Directory where the figure will be saved')
    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--save-name', default=None,
                        help='Base name for the saved figure(s) (e.g., my_analysis). Extension .png will be added. If None, a default name will be used.')
    parser.add_argument('--num-examples-display', type=int, default=20,
                        help='Number of examples to display or process in the final plot (for margin_vs_depth)')
    parser.add_argument('--num-examples-scan', type=int, default=80,
                        help='Total number of diverse examples to scan/collect during data gathering phases (for margin_vs_depth)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for stochastic processes (e.g., data collection sampling)')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='Minimum confidence score for filtering detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching detections to ground truth')
    parser.add_argument('--collection-batch-size', type=int, default=10,
                        help='Batch size to use during the data collection phase')
    parser.add_argument('--collection-max-batches', type=int, default=30,
                        help='Maximum number of batches to process during data collection')
    parser.add_argument('--separate-panels', action='store_true',
                        help='If set, save norm and margin plots as separate images. Otherwise, save as a single subplot figure.')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Always generate both plots, using separate_panels to control output style
    generate_combined_depth_analysis_plot(
        model_path=args.model_path,
        config_path=args.config,
        save_dir=args.save_dir,
        base_save_name=args.save_name,
        separate_panels=args.separate_panels,
        num_examples_to_collect=args.num_examples_scan,
        num_examples_to_process=args.num_examples_display,
        min_score_threshold=args.min_score,
        iou_threshold_for_collection=args.iou_threshold,
        batch_size_for_collection=args.collection_batch_size,
        max_batches_for_collection=args.collection_max_batches,
        random_state_for_collection=args.random_state
    )

if __name__ == '__main__':
    main()
