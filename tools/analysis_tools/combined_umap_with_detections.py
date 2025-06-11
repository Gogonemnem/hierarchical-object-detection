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
import math
import os
import pathlib
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Set
import os
import traceback
import cv2  # type: ignore

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
from matplotlib.axes import Axes
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


def get_last_classification_branch_index_from_state(state_dict):
    """Determine the index of the last classification branch from state_dict."""
    # Find the highest numbered cls_branches in the state_dict
    max_branch_idx = 0
    for key in state_dict.keys():
        if 'bbox_head.cls_branches.' in key:
            # Extract the branch number
            try:
                branch_part = key.split('bbox_head.cls_branches.')[1]
                branch_idx = int(branch_part.split('.')[0])
                max_branch_idx = max(max_branch_idx, branch_idx)
            except (IndexError, ValueError):
                continue
    return max_branch_idx

def get_last_classification_branch_index(model):
    """Determine the index of the last classification branch in the model."""
    for name, module in model.named_modules():
        if hasattr(module, 'num_pred_layer') and 'bbox_head' in name:
            return module.num_pred_layer - 1
    # Fallback to 0 if we can't determine
    return 0

class EmbeddingCollector:
    """
    Collects query embeddings during inference using forward hooks.
    This provides scientifically accurate embeddings by capturing the actual
    feature vectors that get compared to prototype embeddings during detection.
    
    Both prototype and query embeddings come from get_projected_features and are
    in the same transformation space (256-dimensional projected features).
    """
    
    def __init__(self):
        self.embeddings = []
        self.hook_handles = []
        
    def clear(self):
        """Clear collected embeddings for next image"""
        self.embeddings = []
        
    def register_hooks(self, model):
        """Register forward hooks on EmbeddingClassifier modules in the second-to-last classification branch"""
        hooks_registered = 0
        
        # Determine the LAST classification branch index, then use the second-to-last
        # In DINO, all cls_branches[0-5] are decoder layers, and we want the second-to-last (index 4)
        last_branch_idx = get_last_classification_branch_index(model)
        target_branch_idx = last_branch_idx - 1  # Use the SECOND-TO-LAST branch for embedding collection
        target_branch = f'cls_branches.{target_branch_idx}'
        print(f"Targeting classification branch: {target_branch} (branch {target_branch_idx} of {last_branch_idx+1} total - second-to-last)")
        
        # Scan all modules to see what EmbeddingClassifier modules we find
        embedding_classifier_modules = []
        for name, module in model.named_modules():
            if hasattr(module, 'get_projected_features') and 'EmbeddingClassifier' in str(type(module)):
                embedding_classifier_modules.append((name, module))
                print(f"Found EmbeddingClassifier: {name}")
        
        # Now register hooks only on the target branch
        for name, module in model.named_modules():
            if hasattr(module, 'get_projected_features') and target_branch in name and 'EmbeddingClassifier' in str(type(module)):
                print(f"Registering hook on: {name}")
                
                def make_hook(collector_ref, module_name):
                    """Create a hook that captures projected features from get_projected_features"""
                    def hook_fn(module, input, output):
                        # The forward method calls get_projected_features internally
                        # We need to hook the get_projected_features method to capture its output
                        pass  # This forward hook won't directly capture what we need
                    return hook_fn
                
                # Instead of hooking forward, we need to hook get_projected_features method
                # Store original method
                original_method = module.get_projected_features
                
                def make_hooked_method(orig_method, collector_ref, module_name):
                    """Create a hooked version of get_projected_features"""
                    def hooked_method(x):
                        result = orig_method(x)
                        
                        # Store the result which are our query embeddings (256-dim projected features)
                        if hasattr(result, 'shape') and len(result.shape) >= 2:
                            embeddings_batch = result.detach().cpu().numpy()
                            collector_ref.embeddings.append(embeddings_batch)
                        return result
                    return hooked_method
                
                # Replace the method with our hooked version
                module.get_projected_features = make_hooked_method(original_method, self, name)
                
                # Store for cleanup
                self.hook_handles.append((module, 'get_projected_features', original_method))
                hooks_registered += 1
                print(f"Registered method hook on {name}")
                
        print(f"Successfully registered {hooks_registered} hooks")
        return hooks_registered > 0
                
    def remove_hooks(self):
        """Remove all registered hooks and clean up"""
        for module, method_name, original_method in self.hook_handles:
            setattr(module, method_name, original_method)
        self.hook_handles.clear()
        print(f"Restored all original methods")
        
    def get_embeddings_for_detections(self, bbox_index):
        """Extract embeddings corresponding to the detections using bbox_index.
        
        Args:
            bbox_index: Tensor or array of indices indicating which queries from the 
                       original 900 were selected for final predictions
        
        Returns:
            numpy array: Query embeddings corresponding to the selected queries
        """
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
        
        # Convert bbox_index to numpy if it's a tensor
        if hasattr(bbox_index, 'cpu'):
            bbox_index = bbox_index.cpu().numpy()
        
        # Use bbox_index to select the correct embeddings
        # bbox_index contains the indices of queries that were selected for predictions
        try:
            if len(all_embeddings) > max(bbox_index):
                selected_embeddings = all_embeddings[bbox_index]
                return selected_embeddings
            else:
                print(f"Warning: Not enough embeddings ({len(all_embeddings)}) for bbox_index max {max(bbox_index)}")
                return None
        except Exception as e:
            print(f"Error selecting embeddings with bbox_index: {e}")
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


def run_inference_with_hooks(model, dataset, collector, target_examples: int = 20, batch_size: int = 50, min_score: float = 0.25, max_batches: int = 30, hierarchy: HierarchyTree = None, labels: List[str] = None, iou_threshold: float = 0.5):
    """Run inference with forward hooks to collect query embeddings in batches until we have sufficient diversity across all 6 fallback levels."""
    results_with_embeddings = []
    
    # Register hooks
    collector.register_hooks(model)
    
    try:
        img_idx = 0
        batch_count = 0
        
        while batch_count < max_batches and img_idx < len(dataset):
            print(f"Processing batch {batch_count + 1}, images {img_idx}-{min(img_idx + batch_size, len(dataset))}")
            
            # Process batch
            batch_end = min(img_idx + batch_size, len(dataset))
            for current_idx in range(img_idx, batch_end):
                # Clear previous embeddings
                collector.clear()
                
                # Get image info
                img_info = dataset.get_data_info(current_idx)
                img_path = img_info['img_path']
                
                # Run inference - this will trigger our hooks
                with torch.no_grad():
                    result = inference_detector(model, img_path)
                    
                # Extract detection info from DetDataSample
                if hasattr(result, 'pred_instances'):
                    pred_instances = result.pred_instances
                    num_detections = len(pred_instances.bboxes)
                    
                    # Check if we have bbox_index (should be present from EmbeddingDINOHead)
                    if hasattr(pred_instances, 'bbox_index'):
                        bbox_index = pred_instances.bbox_index
                        # print(f"Found bbox_index with {len(bbox_index)} indices: {bbox_index[:10].tolist() if len(bbox_index) > 10 else bbox_index.tolist()}")
                        
                        # Get corresponding query embeddings using bbox_index
                        query_embeddings = collector.get_embeddings_for_detections(bbox_index)
                        
                        if query_embeddings is not None:
                            # Add embeddings to the result
                            pred_instances.query_embeddings = torch.from_numpy(query_embeddings)
                        else:
                            print("Warning: Could not get query embeddings using bbox_index")
                    else:
                        print("Warning: bbox_index not found in pred_instances - falling back to first N embeddings")
                        # Fallback to old method
                        query_embeddings = collector.get_embeddings_for_detections_fallback(num_detections)
                        if query_embeddings is not None:
                            pred_instances.query_embeddings = torch.from_numpy(query_embeddings)
                    
                    results_with_embeddings.append({
                        'image_idx': current_idx,
                        'image_path': img_path,
                        'result': result,
                        'gt_instances': img_info['instances']
                    })
            
            # Check for sufficient diversity across all 6 fallback levels
            if len(results_with_embeddings) >= target_examples * 3:  # Start checking after reasonable amount
                # If we have hierarchical information, do proper diversity check
                if hierarchy is not None and labels is not None:
                    if check_fallback_diversity(results_with_embeddings, min_score, hierarchy, labels, iou_threshold):
                        print(f"Found sufficient diversity across all 6 fallback levels after {len(results_with_embeddings)} results")
                        break
                else:
                    # Fallback to simple high-confidence detection count
                    high_conf_detections = 0
                    for result_data in results_with_embeddings:
                        result = result_data['result']
                        if hasattr(result, 'pred_instances'):
                            scores = result.pred_instances.scores
                            high_conf_detections += len([s for s in scores if s >= min_score])
                    
                    # If we have at least 200 high-confidence detections, likely enough diversity
                    if high_conf_detections >= 200:
                        print(f"Found sufficient diversity after {len(results_with_embeddings)} results")
                        break
            
            img_idx = batch_end
            batch_count += 1
            
            if batch_count % 3 == 0:  # Progress update every 3 batches
                print(f"Processed {len(results_with_embeddings)} results so far...")
                
    finally:
        # Always remove hooks
        collector.remove_hooks()
    
    print(f"Collected {len(results_with_embeddings)} total results from {img_idx} images")
    return results_with_embeddings

def check_fallback_diversity(results_with_embeddings, min_score: float, hierarchy: HierarchyTree, labels: List[str], iou_threshold: float = 0.5) -> bool:
    """Check if we have enough diversity across all 6 fallback levels."""
    level_counts = [0] * 6
    
    for result_data in results_with_embeddings:
        result = result_data['result']
        gt_instances = result_data['gt_instances']
        
        pred_instances = result.pred_instances
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()
        pred_bboxes = pred_instances.bboxes.cpu().numpy()
        
        # Check each high-confidence detection
        for i, (bbox, score, pred_label) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
            if score < min_score:
                continue
                
            # Find matching ground truth
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
                
            # Determine fallback level and count it
            fallback_level = determine_fallback_level(gt_leaf, pred_node, hierarchy)
            level_counts[fallback_level] += 1
    
    # Check if we have at least 2 examples for each of the 6 levels
    min_per_level = 2
    all_levels_covered = all(count >= min_per_level for count in level_counts)
    
    level_names = ['Leaf', 'Parent', 'Grandparent', 'Sibling', 'Cousin', 'Off-branch']
    covered_levels = sum(1 for count in level_counts if count >= min_per_level)
    
    print(f"Fallback level coverage: {covered_levels}/6 levels with {min_per_level}+ examples")
    for i, (name, count) in enumerate(zip(level_names, level_counts)):
        if count > 0:
            print(f"  {name}: {count} examples")
    
    return all_levels_covered


# -----------------------------------------------------------------------------
# Fallback Level Logic (from hierarchical_prediction_distribution.py)
# -----------------------------------------------------------------------------

def determine_fallback_level(gt_leaf: str, pred_node: str, tree: HierarchyTree) -> int:
    """
    Determine the fallback level based on hierarchical relationship.
    
    Args:
        gt_leaf: Ground truth leaf class name
        pred_node: Predicted class name
        tree: Hierarchy tree
        
    Returns:
        int: Fallback level (0=leaf, 1=parent, 2=grandparent, 3=sibling, 4=cousin, 5=off-branch)
    """
    if gt_leaf not in tree.class_to_node or pred_node not in tree.class_to_node:
        return 5  # off-branch
    
    gt_node = tree.class_to_node[gt_leaf]
    
    # 0: Exact leaf match
    if gt_leaf == pred_node:
        return 0
    
    # 1: Parent match
    if gt_node.parent and gt_node.parent.name == pred_node:
        return 1
    
    # 2: Grandparent match
    grandparent = tree.get_grandparent(gt_leaf)
    if grandparent and grandparent == pred_node:
        return 2
    
    # 3: Sibling match
    siblings = tree.get_siblings(gt_leaf)
    if pred_node in siblings:
        return 3
    
    # 4: Cousin match
    cousins = tree.get_cousins(gt_leaf)
    if pred_node in cousins:
        return 4
    
    # 5: Off-branch (no hierarchical relationship)
    return 5


def get_fallback_visual_encoding(level: int) -> Dict[str, Any]:
    """
    Get visual encoding for fallback level.
    Uses colors consistent with hierarchical_prediction_distribution.py
    
    Returns:
        dict: Contains 'color', 'linestyle', 'linewidth' for border
    """
    # Color scheme from hierarchical_prediction_distribution.py for consistency
    encodings = {
        0: {'color': '#2E8B57', 'linestyle': '-', 'linewidth': 4},     # tp (leaf correct) - SeaGreen
        1: {'color': '#66CDAA', 'linestyle': '-', 'linewidth': 4},     # parent_tp - MediumAquamarine
        2: {'color': '#90EE90', 'linestyle': '-', 'linewidth': 4},     # grandparent_tp - LightGreen
        3: {'color': '#87CEEB', 'linestyle': '-', 'linewidth': 4},     # sibling_tp - SkyBlue
        4: {'color': '#DDA0DD', 'linestyle': '-', 'linewidth': 4},     # cousin_tp - Plum
        5: {'color': '#DC143C', 'linestyle': '-', 'linewidth': 4},     # other_class - Crimson (off-branch)
    }
    return encodings.get(level, encodings[5])


def validate_nearest_neighbor_predictions_with_model(model, query_embeddings: np.ndarray, prototype_embeddings: np.ndarray, 
                                                       prediction_labels: np.ndarray, space_name: str = "embedding") -> float:
    """
    Validate that the nearest neighbor in embedding space matches the actual model prediction.
    Uses the model's actual distance calculation with temperature scaling and bias.
    
    Args:
        model: The trained model with classifier
        query_embeddings: Query embeddings (N, D)
        prototype_embeddings: Prototype embeddings (M, D) 
        prediction_labels: Actual prediction labels for each query (N,)
        space_name: Name of the space for logging (e.g., "original", "PCA", "2D")
        
    Returns:
        float: Percentage of queries where nearest neighbor matches prediction
    """
    if len(query_embeddings) == 0 or len(prediction_labels) == 0:
        return 0.0
    
    # Get the target classifier (using same logic as immediate validation)
    last_branch_idx = get_last_classification_branch_index(model)
    is_two_stage = getattr(model, 'as_two_stage', False)
    if is_two_stage:
        target_classifier = model.bbox_head.cls_branches[last_branch_idx - 1]  # Second-to-last decoder layer
    else:
        target_classifier = model.bbox_head.cls_branches[last_branch_idx]
    
    matches = 0
    sample_size = min(10, len(query_embeddings))
    detailed_results = []
    
    with torch.no_grad():
        query_tensor = torch.from_numpy(query_embeddings).float()
        prototype_tensor = torch.from_numpy(prototype_embeddings).float()
        
        for i in range(sample_size):
            # Get single query embedding
            query_emb = query_tensor[i:i+1]  # Shape: (1, D)
            
            # Calculate distances using the model's method
            distances = torch.cdist(query_emb, prototype_tensor, p=2).squeeze(0)  # Shape: (num_prototypes,)
            
            # Apply temperature scaling if present
            if target_classifier.use_temperature and target_classifier.logit_scale is not None:
                temp_scale = target_classifier.logit_scale.exp().clamp(max=100)
                distances = distances * temp_scale
            
            # Convert distances to logits (negative distance)
            logits = -distances
            
            # Apply geometric bias if present
            if target_classifier.geometric_bias is not None:
                logits = logits + target_classifier.geometric_bias
            
            # Find the class with minimum distance (maximum logit)
            nn_idx = torch.argmax(logits).item()
            
            # Check if nearest neighbor matches actual prediction
            pred_label = prediction_labels[i]
            nn_matches_pred = (nn_idx == pred_label)
            
            if nn_matches_pred:
                matches += 1
                
            # Ensure pred_label is an integer for indexing
            try:
                pred_label_int = int(pred_label)
                min_distance = distances[pred_label_int].item()
            except (ValueError, IndexError) as e:
                print(f"Warning: Cannot get distance for pred_label {pred_label}: {e}")
                min_distance = float('nan')
                
            detailed_results.append({
                'query_idx': i,
                'prediction': pred_label,
                'nearest_neighbor': nn_idx,
                'matches': nn_matches_pred,
                'min_distance': min_distance
            })
    
    match_rate = matches / sample_size
    print(f"\n{space_name} space validation (MODEL-AWARE): Nearest neighbor = Prediction")
    print(f"Prediction accuracy via nearest neighbor: {matches}/{sample_size} ({match_rate:.1%})")
    
    # Show detailed breakdown for small samples
    if sample_size <= 5:
        print("Detailed breakdown:")
        for result in detailed_results:
            status = "✅" if result['matches'] else "❌"
            print(f"  Query {result['query_idx']}: Pred={result['prediction']}, NN={result['nearest_neighbor']} {status}")
    
    if match_rate >= 0.9:
        print("✅ Excellent: Nearest neighbor matches prediction (embedding-based classification working)")
    elif match_rate >= 0.7:
        print("⚠️ Good: Most nearest neighbors match predictions")
    elif match_rate >= 0.5:
        print("⚠️ Moderate: Some nearest neighbors match predictions")
    else:
        print("❌ Poor: Nearest neighbors often don't match predictions")
        print("💡 This suggests the embedding space may not be well-aligned for distance-based classification")
    
    return match_rate


def validate_nearest_neighbor_predictions(query_embeddings: np.ndarray, prototype_embeddings: np.ndarray, 
                                         prediction_labels: np.ndarray, space_name: str = "embedding") -> float:
    """
    Validate that the nearest neighbor in embedding space matches the actual model prediction.
    
    Args:
        query_embeddings: Query embeddings (N, D)
        prototype_embeddings: Prototype embeddings (M, D) 
        prediction_labels: Actual prediction labels for each query (N,)
        space_name: Name of the space for logging (e.g., "original", "PCA", "2D")
        
    Returns:
        float: Percentage of queries where nearest neighbor matches prediction
    """
    if len(query_embeddings) == 0 or len(prediction_labels) == 0:
        return 0.0
    
    matches = 0
    sample_size = min(10, len(query_embeddings))
    detailed_results = []
    
    for i in range(sample_size):
        # Find nearest neighbor in embedding space
        query = query_embeddings[i:i+1]
        distances = np.linalg.norm(prototype_embeddings - query, axis=1)
        nn_idx = np.argmin(distances)
        
        # Check if nearest neighbor matches actual prediction
        pred_label = prediction_labels[i]
        nn_matches_pred = (nn_idx == pred_label)
        
        if nn_matches_pred:
            matches += 1
            
        detailed_results.append({
            'query_idx': i,
            'prediction': pred_label,
            'nearest_neighbor': nn_idx,
            'matches': nn_matches_pred,
            'distance': distances[nn_idx]
        })
    
    match_rate = matches / sample_size
    print(f"\n{space_name} space validation: Nearest neighbor = Prediction")
    print(f"Prediction accuracy via nearest neighbor: {matches}/{sample_size} ({match_rate:.1%})")
    
    # Show detailed breakdown for small samples
    if sample_size <= 5:
        print("Detailed breakdown:")
        for result in detailed_results:
            status = "✅" if result['matches'] else "❌"
            print(f"  Query {result['query_idx']}: Pred={result['prediction']}, NN={result['nearest_neighbor']} {status}")
    
    if match_rate >= 0.9:
        print("✅ Excellent: Nearest neighbor matches prediction (embedding-based classification working)")
    elif match_rate >= 0.7:
        print("⚠️ Good: Most nearest neighbors match predictions")
    elif match_rate >= 0.5:
        print("⚠️ Moderate: Some nearest neighbors match predictions")
    else:
        print("❌ Poor: Nearest neighbors often don't match predictions")
        print("💡 This suggests the embedding space may not be well-aligned for distance-based classification")
    
    return match_rate


def collect_embeddings_and_labels_for_validation(results_with_embeddings, min_score: float = 0.3):
    """
    Extract query embeddings and corresponding prediction labels from inference results.
    
    Args:
        results_with_embeddings: Results from run_inference_with_hooks
        min_score: Minimum confidence score for detections to include
        
    Returns:
        tuple: (query_embeddings, prediction_labels) as numpy arrays
    """
    all_query_embeddings = []
    all_prediction_labels = []
    
    for result_data in results_with_embeddings:
        result = result_data['result']
        
        if not hasattr(result, 'pred_instances'):
            continue
            
        pred_instances = result.pred_instances
        
        # Check if we have both embeddings and predictions
        if not hasattr(pred_instances, 'query_embeddings'):
            continue
            
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()
        query_embeddings = pred_instances.query_embeddings.numpy()
        
        # Filter by confidence score
        high_conf_mask = pred_scores >= min_score
        if not np.any(high_conf_mask):
            continue
            
        # Collect high-confidence embeddings and labels
        high_conf_embeddings = query_embeddings[high_conf_mask]
        high_conf_labels = pred_labels[high_conf_mask]
        
        all_query_embeddings.append(high_conf_embeddings)
        all_prediction_labels.append(high_conf_labels)
    
    if not all_query_embeddings:
        return np.array([]), np.array([])
    
    # Concatenate all embeddings and labels
    combined_embeddings = np.vstack(all_query_embeddings)
    combined_labels = np.concatenate(all_prediction_labels)
    
    print(f"Collected {len(combined_embeddings)} query embeddings with prediction labels for validation")
    
    return combined_embeddings, combined_labels


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
    try:
        # Extract annotation file path from config
        print(f"Extracting annotation file path from config: {config_path}")
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        
        ann_file_path = None
        if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
            ann_file = cfg.test_dataloader.dataset.ann_file
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg.test_dataloader.dataset, 'data_root') else ''
            if data_root and not os.path.isabs(ann_file):
                ann_file_path = os.path.join(data_root, ann_file)
            else:
                ann_file_path = ann_file
        
        # Fallback to a reasonable default if not found
        if ann_file_path is None:
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset') and hasattr(cfg.test_dataloader.dataset, 'data_root') else 'data/aircraft'
            ann_file_path = os.path.join(data_root, 'aircraft_test.json')
            print(f"Warning: Could not find annotation file in config, using fallback: {ann_file_path}")
        
        print(f"Using annotation file: {ann_file_path}")
        
        model_path_obj = pathlib.Path(model_path)
        ann_file_path_obj = pathlib.Path(ann_file_path)

        if not model_path_obj.exists():
            print(f"Error: Model file not found at '{model_path_obj}'")
            return None, None, None, None, None, None, None
        if not ann_file_path_obj.exists():
            print(f"Error: Annotation file not found at '{ann_file_path_obj}'")
            return None, None, None, None, None, None, None

        # Always load the model for validation purposes
        print(f"Loading model from {model_path}")
        
        # Determine config path from model path
        actual_config_path = str(model_path_obj).replace('.pth', '.py')
        if not pathlib.Path(actual_config_path).exists():
            print(f"Warning: Config file not found at {actual_config_path}, trying to infer...")
            # Try to find config in the same directory
            model_dir = model_path_obj.parent
            config_files = list(model_dir.glob("*.py"))
            if config_files:
                actual_config_path = str(config_files[0])
                print(f"Using config file: {actual_config_path}")
            else:
                raise FileNotFoundError(f"Could not find config file for model at {model_path}")
        
        # Load model
        model = init_detector(actual_config_path, model_path, device='cpu')

        # Use pre-computed embeddings if provided, otherwise extract prototypes from model
        if prototype_embeddings is not None:
            embeddings = prototype_embeddings
            print(f"Using pre-computed prototype embeddings with shape {embeddings.shape}")
        else:
            # Use the model's prototypes property to get properly transformed embeddings
            # This ensures consistency with query embeddings captured from get_projected_features hook
            last_branch_idx = get_last_classification_branch_index(model)
            target_branch_idx = last_branch_idx - 1  # Second-to-last branch for actual classification
            target_classifier = getattr(model.bbox_head.cls_branches, str(target_branch_idx))
            embeddings = target_classifier.prototypes.detach().cpu().numpy()
            print(f"Using model's prototypes property from branch {target_branch_idx}")
            print(f"Extracted {len(embeddings)} prototype embeddings with shape {embeddings.shape}")

        # Load annotation data
        ann = load(ann_file_path_obj)
        if not isinstance(ann, dict) or "categories" not in ann or "taxonomy" not in ann:
            print("Error: Annotation file is missing 'categories' or 'taxonomy' keys.")
            return None, None, None, None, None, None, None
        
        categories = ann["categories"]
        if len(embeddings) != len(categories):
            print(f"Warning: Number of embeddings ({len(embeddings)}) != categories ({len(categories)})")

        # Prepare embeddings for UMAP fitting
        query_projections = None
        if query_embeddings is not None and len(query_embeddings) > 0:
            print(f"Including {len(query_embeddings)} query embeddings in UMAP fitting...")
            
            # Verify query embeddings are the same dimensionality as prototypes
            if query_embeddings.shape[1] != embeddings.shape[1]:
                print(f"Warning: Query embedding dimensionality ({query_embeddings.shape[1]}) " +
                      f"doesn't match prototype dimensionality ({embeddings.shape[1]}). Skipping.")
                combined_embeddings = embeddings
            else:
                # Print some stats to verify the embeddings look reasonable
                proto_norm = np.mean(np.linalg.norm(embeddings, axis=1))
                query_norm = np.mean(np.linalg.norm(query_embeddings, axis=1))
                print(f"Prototype embeddings mean norm: {proto_norm:.4f}")
                print(f"Query embeddings mean norm: {query_norm:.4f}")
                
                # DEBUG: Check query embedding diversity
                if len(query_embeddings) > 1:
                    from scipy.spatial.distance import pdist
                    query_distances = pdist(query_embeddings, metric='euclidean')
                    mean_distance = np.mean(query_distances)
                    print(f"Query embedding diversity - mean distance: {mean_distance:.6f}")
                    
                    # Check if queries are too similar
                    if mean_distance < 0.01:
                        print(f"WARNING: Query embeddings are very similar (mean distance: {mean_distance:.6f})")
                        print(f"         This could cause MDS to place them all at the same location")
                
                # Combine prototype and query embeddings for joint UMAP fitting
                combined_embeddings = np.vstack([embeddings, query_embeddings])
        else:
            combined_embeddings = embeddings

        # Fit UMAP on combined embeddings with optimized settings for joint fitting
        if query_embeddings is not None and len(query_embeddings) > 0:
            # Use more conservative settings when fitting both prototypes and queries together
            n_neighbors = min(10, len(combined_embeddings)-1 if len(combined_embeddings) > 1 else 1)
            min_dist = 0.05  # Tighter clusters for better prototype-query relationships
            print(f"Joint UMAP fitting with {len(embeddings)} prototypes + {len(query_embeddings)} queries")
            print(f"Using n_neighbors={n_neighbors}, min_dist={min_dist} for better query positioning")
        else:
            # Standard settings for prototype-only fitting
            n_neighbors = min(15, len(combined_embeddings)-1 if len(combined_embeddings) > 1 else 1)
            min_dist = 0.1
            print(f"Prototype-only UMAP fitting with {len(embeddings)} prototypes")

        # === PRE-TRANSFORMATION VALIDATION ===
        # Validate that nearest neighbors match actual predictions in the original embedding space
        print("\n=== Pre-transformation Validation (Original Embedding Space) ===")
        if query_embeddings is not None and len(query_embeddings) > 0:
            if detection_examples is not None and len(detection_examples) > 0:
                # Extract prediction labels (class indices) directly from detection examples
                validation_pred_labels = np.array([ex['pred_label'] for ex in detection_examples])
                validation_queries = query_embeddings  # These should match the detection examples
                
                if len(validation_queries) == len(validation_pred_labels):
                    # Validate that nearest neighbors match actual predictions in original space
                    original_match_rate = validate_nearest_neighbor_predictions_with_model(
                        model, validation_queries, embeddings, validation_pred_labels, 
                        "Original embedding"
                    )
                    print(f"Baseline validation complete: {original_match_rate:.1%} of queries have nearest neighbor = prediction")
                else:
                    print(f"⚠️  Mismatch: {len(validation_queries)} query embeddings vs {len(validation_pred_labels)} prediction labels")
                    print("    Cannot perform pre-transformation validation")
            else:
                print("⚠️  No detection examples available for validation")
                print("    Pre-transformation validation requires detection examples with prediction labels")
        else:
            print("⚠️  No query embeddings available for validation")

        # Apply PCA preprocessing if sklearn is available and we have enough samples
        # CRITICAL: PCA must be fitted on the combined embeddings (prototypes + queries) 
        # to ensure both are in the same preprocessed space
        preprocessed_embeddings = combined_embeddings
        pca_transformer = None  # Store the PCA transformer for validation
        
        if PCA_AVAILABLE and len(combined_embeddings) > 50:
            n_components = min(50, combined_embeddings.shape[1], len(combined_embeddings) - 1)
            if n_components > 10:  # Only apply if we can meaningfully reduce dimensionality
                print(f"Applying PCA preprocessing: {combined_embeddings.shape[1]}D → {n_components}D")
                print(f"PCA fitted on {len(combined_embeddings)} total embeddings (prototypes + queries)")
                
                pca_transformer = PCA(n_components=n_components, random_state=random_state)
                preprocessed_embeddings = pca_transformer.fit_transform(combined_embeddings)
                explained_var = np.sum(pca_transformer.explained_variance_ratio_)
                print(f"PCA preserved {explained_var:.1%} of variance")
                
                # Validate nearest neighbor preservation after PCA
                print("\n=== PCA Nearest Neighbor Validation ===")
                if query_embeddings is not None and len(query_embeddings) > 0:
                    # Split back to original and preprocessed for comparison
                    original_prototypes = embeddings
                    original_queries = query_embeddings
                    preprocessed_prototypes = preprocessed_embeddings[:len(embeddings)]
                    preprocessed_queries = preprocessed_embeddings[len(embeddings):]
                    
                    # Check nearest neighbor preservation for a sample of queries
                    sample_size = min(10, len(original_queries))
                    preserved_nn = 0
                    pca_detailed_results = []
                    
                    print("Validating basic nearest neighbor preservation (without prediction verification):")
                    
                    for i in range(sample_size):
                        # Find nearest neighbor in original space
                        query_orig = original_queries[i:i+1]
                        distances_orig = np.linalg.norm(original_prototypes - query_orig, axis=1)
                        nn_orig = np.argmin(distances_orig)
                        
                        # Find nearest neighbor in PCA space
                        query_pca = preprocessed_queries[i:i+1]
                        distances_pca = np.linalg.norm(preprocessed_prototypes - query_pca, axis=1)
                        nn_pca = np.argmin(distances_pca)
                        
                        preserved = nn_orig == nn_pca
                        if preserved:
                            preserved_nn += 1
                        
                        pca_detailed_results.append({
                            'query_idx': i,
                            'nn_orig': nn_orig,
                            'nn_pca': nn_pca,
                            'preserved': preserved,
                            'orig_dist': distances_orig[nn_orig],
                            'pca_dist': distances_pca[nn_pca]
                        })
                    
                    nn_preservation = preserved_nn / sample_size
                    print(f"Nearest neighbor preservation: {preserved_nn}/{sample_size} ({nn_preservation:.1%})")
                    
                    # Show detailed breakdown for debugging
                    if sample_size <= 5:  # Only show details for small samples
                        print("Detailed breakdown:")
                        for result in pca_detailed_results:
                            status = "✅" if result['preserved'] else "❌"
                            print(f"  Query {result['query_idx']}: NN {result['nn_orig']} → {result['nn_pca']} {status}")
                    
                    if nn_preservation >= 0.8:
                        print("✅ Good: PCA preserves nearest neighbor relationships")
                    elif nn_preservation >= 0.6:
                        print("⚠️ Moderate: Some nearest neighbor relationships changed")
                    else:
                        print("❌ Poor: Many nearest neighbor relationships changed")
                    
                    # Additional validation with actual predictions if detection examples are available
                    if detection_examples is not None and len(detection_examples) > 0:
                        # Extract prediction labels (class indices) directly from detection examples
                        validation_pred_labels = np.array([ex['pred_label'] for ex in detection_examples])
                        
                        if len(validation_pred_labels) == len(preprocessed_queries):
                            # Note: For PCA validation, we use simple L2 distance since the model doesn't operate in PCA space
                            pca_match_rate = validate_nearest_neighbor_predictions(
                                preprocessed_queries, preprocessed_prototypes, validation_pred_labels, 
                                "PCA embedding"
                            )
                            print(f"PCA validation: {pca_match_rate:.1%} of queries have nearest neighbor = prediction")
                        else:
                            print("⚠️ Cannot validate PCA predictions: embedding/label count mismatch")
                    else:
                        print("💡 Note: To validate that nearest neighbors = predictions, detection examples needed")
                
                # Validate that both prototypes and queries were preprocessed consistently
                if query_embeddings is not None and len(query_embeddings) > 0:
                    print(f"✅ Both prototypes and queries preprocessed by same PCA transformation")
                    print(f"   Preprocessed prototypes: {len(embeddings)} samples")
                    print(f"   Preprocessed queries: {len(query_embeddings)} samples")
                print("=" * 40)
            else:
                print("Skipping PCA: insufficient dimensionality reduction benefit")
        elif not PCA_AVAILABLE:
            print("Scikit-learn not available, skipping PCA preprocessing")
        else:
            print("Skipping PCA: insufficient samples for preprocessing")

        # Choose projection method: UMAP or MetricMDS
        if use_mds and PCA_AVAILABLE:
            print("Using MetricMDS for better distance preservation...")
            mds = MDS(n_components=2, metric=True, random_state=random_state, 
                     dissimilarity='euclidean', normalized_stress='auto')
            combined_umap_2d = mds.fit_transform(preprocessed_embeddings)
            reducer = None  # MDS doesn't return a reusable transformer
            print(f"MDS fitting completed. Output shape: {combined_umap_2d.shape}")
            print(f"MDS stress: {mds.stress_:.4f}")
            
            # Check for potential issues with MDS output
            if query_embeddings is not None:
                query_coords = combined_umap_2d[len(embeddings):]
                query_variance_x = np.var(query_coords[:, 0])
                query_variance_y = np.var(query_coords[:, 1])
                if query_variance_x < 1e-6 and query_variance_y < 1e-6:
                    print(f"WARNING: Query coordinates have very low variance (X: {query_variance_x:.8f}, Y: {query_variance_y:.8f})")
                    print("This suggests the query embeddings are too similar or there's a bug in the embedding collection")
        else:
            # Use UMAP with optimized settings
            if query_embeddings is not None and len(query_embeddings) > 0:
                # Use more conservative settings when fitting both prototypes and queries together
                n_neighbors = min(10, len(combined_embeddings)-1 if len(combined_embeddings) > 1 else 1)
                min_dist = 0.05  # Tighter clusters for better prototype-query relationships
                print(f"Joint UMAP fitting with {len(embeddings)} prototypes + {len(query_embeddings)} queries")
                print(f"Using n_neighbors={n_neighbors}, min_dist={min_dist} for better query positioning")
            else:
                # Standard settings for prototype-only fitting
                n_neighbors = min(15, len(combined_embeddings)-1 if len(combined_embeddings) > 1 else 1)
                min_dist = 0.1
                print(f"Prototype-only UMAP fitting with {len(embeddings)} prototypes")
                
            reducer = umap.UMAP(
                random_state=random_state, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                metric='euclidean',
                spread=1.0
            )
            print(f"Fitting UMAP on {len(preprocessed_embeddings)} total embeddings...")
            combined_umap_2d = reducer.fit_transform(preprocessed_embeddings)
            print(f"UMAP fitting completed. Output shape: {combined_umap_2d.shape}")
        
        # Ensure we have a proper numpy array
        if hasattr(combined_umap_2d, 'toarray'):  # For sparse matrix
            combined_umap_2d = combined_umap_2d.toarray()
        else:  # Already dense
            combined_umap_2d = np.asarray(combined_umap_2d)
        
        # Validate nearest neighbor preservation after dimensionality reduction (UMAP/MDS)
        if query_embeddings is not None and len(query_embeddings) > 0:
            print(f"\n=== {'MDS' if use_mds and PCA_AVAILABLE else 'UMAP'} Nearest Neighbor Validation ===")
            
            # Split 2D projections
            prototype_2d = combined_umap_2d[:len(embeddings)]
            query_2d = combined_umap_2d[len(embeddings):]
            
            # Check nearest neighbor preservation from high-D to 2D
            sample_size = min(10, len(query_embeddings))
            preserved_nn_2d = 0
            detailed_results = []
            
            for i in range(sample_size):
                # Find nearest neighbor in high-D space (preprocessed embeddings)
                query_high_d = preprocessed_embeddings[len(embeddings) + i:len(embeddings) + i + 1]
                distances_high_d = np.linalg.norm(preprocessed_embeddings[:len(embeddings)] - query_high_d, axis=1)
                nn_high_d = np.argmin(distances_high_d)
                
                # Find nearest neighbor in 2D space
                query_2d_point = query_2d[i:i+1]
                distances_2d = np.linalg.norm(prototype_2d - query_2d_point, axis=1)
                nn_2d = np.argmin(distances_2d)
                
                preserved = nn_high_d == nn_2d
                if preserved:
                    preserved_nn_2d += 1
                
                # Store detailed results for debugging
                detailed_results.append({
                    'query_idx': i,
                    'nn_high_d': nn_high_d,
                    'nn_2d': nn_2d,
                    'preserved': preserved,
                    'high_d_dist': distances_high_d[nn_high_d],
                    '2d_dist': distances_2d[nn_2d]
                })
            
            nn_preservation_2d = preserved_nn_2d / sample_size
            print(f"High-D to 2D nearest neighbor preservation: {preserved_nn_2d}/{sample_size} ({nn_preservation_2d:.1%})")
            
            # Show detailed breakdown for debugging
            if sample_size <= 5:  # Only show details for small samples
                print("Detailed breakdown:")
                for result in detailed_results:
                    status = "✅" if result['preserved'] else "❌"
                    print(f"  Query {result['query_idx']}: NN {result['nn_high_d']} → {result['nn_2d']} {status}")
            
            if nn_preservation_2d >= 0.8:
                print("✅ Excellent: 2D projection preserves nearest neighbor relationships")
            elif nn_preservation_2d >= 0.6:
                print("⚠️ Good: Most nearest neighbor relationships preserved in 2D")
            elif nn_preservation_2d >= 0.4:
                print("⚠️ Moderate: Some nearest neighbor relationships changed in 2D")
            else:
                print("❌ Poor: Many nearest neighbor relationships lost in 2D projection")
                print("💡 Consider: Different UMAP parameters, MetricMDS, or more PCA components")
            
            # Additional validation: Check if original space nearest neighbors are also preserved
            if pca_transformer is not None:
                print("\nOriginal to 2D nearest neighbor preservation:")
                preserved_nn_orig_2d = 0
                orig_detailed_results = []
                
                for i in range(sample_size):
                    # Find nearest neighbor in original space
                    query_orig = query_embeddings[i:i+1]
                    distances_orig = np.linalg.norm(embeddings - query_orig, axis=1)
                    nn_orig = np.argmin(distances_orig)
                    
                    # Find nearest neighbor in 2D space (same as above)
                    query_2d_point = query_2d[i:i+1]
                    distances_2d = np.linalg.norm(prototype_2d - query_2d_point, axis=1)
                    nn_2d = np.argmin(distances_2d)
                    
                    preserved = nn_orig == nn_2d
                    if preserved:
                        preserved_nn_orig_2d += 1
                    
                    orig_detailed_results.append({
                        'query_idx': i,
                        'nn_orig': nn_orig,
                        'nn_2d': nn_2d,
                        'preserved': preserved
                    })
                
                nn_preservation_orig_2d = preserved_nn_orig_2d / sample_size
                print(f"Original to 2D nearest neighbor preservation: {preserved_nn_orig_2d}/{sample_size} ({nn_preservation_orig_2d:.1%})")
                
                # Show detailed breakdown for debugging
                if sample_size <= 5:  # Only show details for small samples
                    print("Detailed breakdown:")
                    for result in orig_detailed_results:
                        status = "✅" if result['preserved'] else "❌"
                        print(f"  Query {result['query_idx']}: NN {result['nn_orig']} → {result['nn_2d']} {status}")
                
                if nn_preservation_orig_2d >= 0.7:
                    print("✅ Great: End-to-end nearest neighbor preservation")
                elif nn_preservation_orig_2d >= 0.5:
                    print("⚠️ Moderate: Some end-to-end nearest neighbor preservation")
                else:
                    print("❌ Poor: Limited end-to-end nearest neighbor preservation")
            
            # Additional validation with actual predictions if detection examples are available
            if detection_examples is not None and len(detection_examples) > 0:
                # Extract prediction labels (class indices) directly from detection examples
                validation_pred_labels = np.array([ex['pred_label'] for ex in detection_examples])
                
                if len(validation_pred_labels) == len(query_2d):
                    print("\nValidating 2D projection with actual predictions:")
                    projection_2d_prototypes = combined_umap_2d[:len(embeddings)]
                    projection_2d_queries = combined_umap_2d[len(embeddings):]
                    
                    # Note: For 2D validation, we use simple L2 distance since the model doesn't operate in 2D space
                    projection_match_rate = validate_nearest_neighbor_predictions(
                        projection_2d_queries, projection_2d_prototypes, validation_pred_labels, 
                        "2D projection"
                    )
                    print(f"Final 2D validation: {projection_match_rate:.1%} of queries have nearest neighbor = prediction")
                    
                    # Also validate the high-D preprocessed space if available
                    if pca_transformer is not None:
                        preprocessed_prototypes_2d = preprocessed_embeddings[:len(embeddings)]
                        preprocessed_queries_2d = preprocessed_embeddings[len(embeddings):]
                        
                        # Note: For high-D validation, we use simple L2 distance since the model doesn't operate in PCA space
                        high_d_match_rate = validate_nearest_neighbor_predictions(
                            preprocessed_queries_2d, preprocessed_prototypes_2d, validation_pred_labels, 
                            "High-D preprocessed"
                        )
                        print(f"High-D preprocessed validation: {high_d_match_rate:.1%} of queries have nearest neighbor = prediction")
                else:
                    print(f"⚠️ Cannot validate 2D predictions: {len(validation_pred_labels)} labels vs {len(query_2d)} queries")
            else:
                print("💡 Note: Detection examples needed for prediction validation")
            
            print("=" * 50)
        
        # Split projections back into prototype and query parts
        umap_2d = combined_umap_2d[:len(embeddings)]
        if query_embeddings is not None and len(query_embeddings) > 0:
            query_projections = combined_umap_2d[len(embeddings):]
            print(f"Split UMAP results: {len(umap_2d)} prototype projections, {len(query_projections)} query projections")
            
            # VALIDATION: Ensure preprocessing consistency
            if pca_transformer is not None:
                # Split the preprocessed embeddings to validate the preprocessing was applied correctly
                preprocessed_prototypes = preprocessed_embeddings[:len(embeddings)]
                preprocessed_queries = preprocessed_embeddings[len(embeddings):]
                
                print(f"✅ PCA Preprocessing Validation:")
                print(f"   Original prototype shape: {embeddings.shape}")
                print(f"   Original query shape: {query_embeddings.shape}")
                print(f"   Preprocessed prototype shape: {preprocessed_prototypes.shape}")
                print(f"   Preprocessed query shape: {preprocessed_queries.shape}")
                
                # Check that both have the same number of components
                if preprocessed_prototypes.shape[1] == preprocessed_queries.shape[1]:
                    print(f"   ✅ Both use same {preprocessed_prototypes.shape[1]} PCA components")
                else:
                    print(f"   ❌ Dimensionality mismatch: prototypes={preprocessed_prototypes.shape[1]}, queries={preprocessed_queries.shape[1]}")
                    
                # Check that the preprocessing applied meaningful transformation
                proto_var_reduction = (embeddings.shape[1] - preprocessed_prototypes.shape[1]) / embeddings.shape[1]
                print(f"   Dimensionality reduction: {proto_var_reduction:.1%} ({embeddings.shape[1]}D → {preprocessed_prototypes.shape[1]}D)")
            else:
                print(f"✅ No PCA applied - prototypes and queries used in original {embeddings.shape[1]}D space")
            
            # Compute projection quality metrics if we have enough data points
            if SCIPY_AVAILABLE and len(combined_embeddings) >= 10:
                print("\n=== Projection Quality Diagnostics ===")
                try:
                    # Compute trustworthiness (how well nearest neighbors are preserved)
                    from sklearn.manifold import trustworthiness
                    trust_score = trustworthiness(preprocessed_embeddings, combined_umap_2d, n_neighbors=min(5, len(combined_embeddings)-1))
                    print(f"Trustworthiness (k=5): {trust_score:.4f} {'✅ Good' if trust_score > 0.87 else '⚠️ Moderate' if trust_score > 0.7 else '❌ Poor'}")
                    
                    # Compute distance correlation between high-D and low-D spaces
                    from scipy.spatial.distance import pdist
                    from scipy.stats import pearsonr
                    high_d_distances = pdist(preprocessed_embeddings, metric='euclidean')
                    low_d_distances = pdist(combined_umap_2d, metric='euclidean')
                    distance_corr, _ = pearsonr(high_d_distances, low_d_distances)
                    print(f"Distance correlation: {distance_corr:.4f} {'✅ Good' if distance_corr > 0.6 else '⚠️ Moderate' if distance_corr > 0.3 else '❌ Poor'}")
                    
                    # Compute continuity (how well local structure is preserved)
                    from sklearn.neighbors import NearestNeighbors
                    k = min(5, len(combined_embeddings) - 1)
                    nbrs_high = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(preprocessed_embeddings)
                    nbrs_low = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(combined_umap_2d)
                    
                    _, indices_high = nbrs_high.kneighbors(preprocessed_embeddings)
                    _, indices_low = nbrs_low.kneighbors(combined_umap_2d)
                    
                    # Calculate continuity: how many high-D neighbors remain neighbors in low-D
                    continuity_scores = []
                    for i in range(len(preprocessed_embeddings)):
                        high_neighbors = set(indices_high[i][1:])  # Exclude self
                        low_neighbors = set(indices_low[i][1:])    # Exclude self
                        overlap = len(high_neighbors.intersection(low_neighbors))
                        continuity_scores.append(overlap / k)
                    
                    continuity = np.mean(continuity_scores)
                    print(f"Continuity (k={k}): {continuity:.4f} {'✅ Good' if continuity > 0.7 else '⚠️ Moderate' if continuity > 0.5 else '❌ Poor'}")
                    
                    # Overall assessment
                    if trust_score > 0.87 and distance_corr > 0.6:
                        print("📊 Overall: Excellent projection quality")
                    elif trust_score > 0.75 and distance_corr > 0.4:
                        print("📊 Overall: Good projection quality")
                    elif trust_score > 0.65 and distance_corr > 0.25:
                        print("📊 Overall: Moderate projection quality - some distortion expected")
                    else:
                        print("📊 Overall: Poor projection quality - significant distortion present")
                        print("💡 Consider: More PCA components, different UMAP parameters, or MetricMDS")
                        
                except ImportError as e:
                    print(f"Some diagnostics unavailable: {e}")
                except Exception as e:
                    print(f"Error computing diagnostics: {e}")
                print("=" * 40)
            
            # Verify the query projections are reasonable
            if len(query_projections) > 0:
                proto_x_range = (np.min(umap_2d[:, 0]), np.max(umap_2d[:, 0]))
                proto_y_range = (np.min(umap_2d[:, 1]), np.max(umap_2d[:, 1]))
                query_x_range = (np.min(query_projections[:, 0]), np.max(query_projections[:, 0]))
                query_y_range = (np.min(query_projections[:, 1]), np.max(query_projections[:, 1]))
                
                print(f"Prototype UMAP range: X={proto_x_range}, Y={proto_y_range}")
                print(f"Query UMAP range: X={query_x_range}, Y={query_y_range}")
        else:
            query_projections = None

        # Build hierarchy and hue map first to get all node names
        categories = ann["categories"]
        hierarchy = HierarchyTree(ann["taxonomy"])
        hue_map = build_hue_map(hierarchy.root)
        
        # Get ALL hierarchy node names (both leaf and parent nodes)
        all_hierarchy_nodes = []
        def collect_nodes(node):
            all_hierarchy_nodes.append(node.name)
            for child in node.children:
                collect_nodes(child)
        collect_nodes(hierarchy.root)
        
        # Use ALL embeddings (116) and map them to ALL hierarchy nodes
        if len(embeddings) != len(all_hierarchy_nodes):
            print(f"Using all {len(embeddings)} embeddings for {len(all_hierarchy_nodes)} hierarchy nodes")
        
        # The labels should be ALL hierarchy nodes, not just categories
        labels = all_hierarchy_nodes
        
        # Create coordinate mapping for ALL nodes that have embeddings
        node_name_to_umap_coords = {}
        for i, name in enumerate(labels):
            if i < len(umap_2d) and name in hierarchy.class_to_node:
                node_name_to_umap_coords[name] = umap_2d[i]

        return embeddings, reducer, hierarchy, hue_map, labels, node_name_to_umap_coords, query_projections

    except Exception as e:
        print(f"Error in load_embeddings_and_umap: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None


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


def plot_prototype_scatter(ax: Axes, 
                          filtered_coords: np.ndarray, 
                          marker_sizes: np.ndarray, 
                          fill_colors: List[Any], 
                          edge_colors: List[Any], 
                          filtered_labels: List[str]):
    """Plot UMAP scatter points with improved label readability."""
    if filtered_coords.size == 0 or marker_sizes.size == 0:
        return

    # Use larger markers for better visibility in publication
    marker_sizes = marker_sizes * 1.3  # Increase marker sizes
    
    # Plot main scatter points
    ax.scatter(
        filtered_coords[:, 0], filtered_coords[:, 1],
        s=marker_sizes,
        c=fill_colors,
        edgecolors=edge_colors,
        linewidths=1.8,
        alpha=0.9,      # More solid appearance
        zorder=2
    )

    # Add labels with better placement and readability
    texts = []
    for i, ((x, y), lbl) in enumerate(zip(filtered_coords, filtered_labels)):
        current_marker_size = marker_sizes[i] if i < len(marker_sizes) else 20
        
        # Calculate font size based on marker size but with limits for readability
        font_size = max(8, min(12, 11 - 0.04 * np.sqrt(current_marker_size)))
        edge_color = edge_colors[i] if i < len(edge_colors) else 'black'

        # Truncate very long labels
        display_label = lbl
        if len(lbl) > 15:
            display_label = lbl[:12] + "..."

        # Add simple text labels without background boxes
        texts.append(ax.text(
            x, y, display_label,
            color=edge_color, 
            fontsize=font_size, 
            fontweight='bold',
            ha="center", va="center",
            alpha=0.9,
            zorder=3
        ))
    
    # Use text adjustment to prevent overlapping WITHOUT arrows
    if texts:
        try:
            # Configure adjustment parameters for cleaner results with stronger separation
            adjust_text(texts, ax=ax, 
                       arrowprops=None,  # Remove arrows completely
                       expand_points=(2.5, 2.5),  # More space between points and text
                       force_text=(1.2, 1.2),     # Stronger text separation force
                       force_points=(0.3, 0.3))   # Moderate point repulsion
        except Exception as e:
            print(f"Could not adjust texts: {e}")
            # Fallback to simple white outline if adjust_text fails
            try:
                import matplotlib.patheffects as path_effects
                for text in texts:
                    text.set_path_effects([
                        path_effects.withStroke(linewidth=2, foreground='white')
                    ])
            except Exception:
                print("Could not apply path effects either")


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
    
    # Run inference with hooks to collect query embeddings
    results_with_embeddings = run_inference_with_hooks(
        model, dataset, collector, target_examples=num_examples, batch_size=50, min_score=min_score, 
        hierarchy=hierarchy, labels=labels, iou_threshold=iou_threshold)
    
    if not results_with_embeddings:
        return []
    
    # DEBUG: Check prototype embeddings characteristics
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
        
        # IMMEDIATE VALIDATION: Check if collected embeddings match predictions using MODEL's distance calculation
        # if len(query_embeddings) > 0 and len(pred_labels) > 0:
        #     print(f"\n=== IMMEDIATE VALIDATION (at collection point) ===")
        #     print(f"Image: {img_path}")
        #     print(f"Query embeddings shape: {query_embeddings.shape}")
        #     print(f"Prototype embeddings shape: {prototype_embeddings.shape}")
        #     print(f"Pred labels: {pred_labels[:min(5, len(pred_labels))]}")  # Show first 5
        #     print(f"Scores: {pred_scores[:min(5, len(pred_scores))]}")
            
        #     # Get the target classifier to use its distance calculation (includes temperature scaling!)
        #     # CRITICAL FIX: For deformable DETR with as_two_stage=True:
        #     # - The LAST layer (index 6) is used for encoder proposal generation
        #     # - The SECOND-TO-LAST layer (index 5) is used for final decoder predictions!
        #     last_branch_idx = get_last_classification_branch_index(model)
            
        #     # CRITICAL FIX: For DINO two-stage models:
        #     # - cls_branches[6] is used for encoder proposal generation  
        #     # - cls_branches[5] is used for actual decoder inference (which predict_by_feat uses)
        #     # The inference always uses the LAST decoder layer, not the encoder proposal layer
        #     is_two_stage = getattr(model, 'as_two_stage', False)
        #     if is_two_stage:
        #         # For two-stage DINO: inference uses the LAST decoder layer (second-to-last branch)
        #         target_branch_idx = last_branch_idx - 1
        #         stage_info = f"decoder layer {target_branch_idx} (actual inference path for two-stage)"
        #     else:
        #         # For single-stage: use last layer for final predictions  
        #         target_branch_idx = last_branch_idx
        #         stage_info = f"decoder layer {target_branch_idx} (single-stage mode)"
                
        #     target_classifier = getattr(model.bbox_head.cls_branches, str(target_branch_idx))
            
        #     print(f"    DEBUG: Total branches={last_branch_idx+1}, Two-stage={is_two_stage}, Using ACTUAL inference branch={target_branch_idx} ({stage_info})")
            
        #     # Test using the EXACT same distance calculation as the model
        #     sample_size = min(3, len(query_embeddings), len(pred_labels))
        #     for test_i in range(sample_size):
        #         query = query_embeddings[test_i:test_i+1]
        #         pred_label = pred_labels[test_i]
        #         score = pred_scores[test_i]
                
        #         # Manual distance calculation (what we were using)
        #         manual_distances = np.linalg.norm(prototype_embeddings - query, axis=1)
        #         manual_nn_idx = np.argmin(manual_distances)
                
        #         # Model's ACTUAL distance calculation (includes temperature scaling!)
        #         query_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)  # (1, 1, D) - add batch and query dims
        #         prototype_tensor = torch.tensor(prototype_embeddings, dtype=torch.float32)  # (M, D)
                
        #         # Move to same device as model
        #         device = next(target_classifier.parameters()).device
        #         query_tensor = query_tensor.to(device)
        #         prototype_tensor = prototype_tensor.to(device)
                
        #         # Test final classification branch
        #         with torch.no_grad():
        #             # Get raw distance logits
        #             distance_logits = target_classifier.get_distance_logits(query_tensor, prototype_tensor)
        #             model_distances = -distance_logits.squeeze().cpu().numpy()
                    
        #             # Apply bias if it exists
        #             final_logits = distance_logits.squeeze()
        #             if target_classifier.geometric_bias is not None:
        #                 final_logits = final_logits + target_classifier.geometric_bias
                    
        #             # Get final prediction using logits (before sigmoid)
        #             logit_based_prediction = torch.argmax(final_logits).item()
                    
        #             # Also test sigmoid-based prediction (what the inference code might use)
        #             sigmoid_probs = torch.sigmoid(final_logits)
        #             sigmoid_based_prediction = torch.argmax(sigmoid_probs).item()
                
        #         model_nn_idx = np.argmin(model_distances)
                
        #         print(f"  Query {test_i}: Pred={pred_label}, Manual_NN={manual_nn_idx}, Model_NN={model_nn_idx}, Score={score:.4f}")
        #         print(f"    Pred class: {labels[pred_label] if pred_label < len(labels) else 'INVALID'}")
        #         print(f"    Manual NN class: {labels[manual_nn_idx] if manual_nn_idx < len(labels) else 'INVALID'}")
        #         print(f"    Model NN class: {labels[model_nn_idx] if model_nn_idx < len(labels) else 'INVALID'}")
        #         print(f"    Logit-based pred: {labels[logit_based_prediction] if logit_based_prediction < len(labels) else 'INVALID'}")
        #         print(f"    Sigmoid-based pred: {labels[sigmoid_based_prediction] if sigmoid_based_prediction < len(labels) else 'INVALID'}")
                
        #         # Show distance differences
        #         print(f"    Manual distance to pred: {manual_distances[pred_label]:.4f}")
        #         print(f"    Model distance to pred: {model_distances[pred_label]:.4f}")
                
        #         # Show which method matches the actual prediction
        #         logit_matches = (logit_based_prediction == pred_label)
        #         sigmoid_matches = (sigmoid_based_prediction == pred_label)
        #         model_nn_matches = (model_nn_idx == pred_label)
        #         print(f"    Logit-based matches: {logit_matches}, Sigmoid-based matches: {sigmoid_matches}, Model NN matches: {model_nn_matches}")
                
        #         # Show bias if it exists
        #         if target_classifier.geometric_bias is not None:
        #             print(f"    Geometric bias: {target_classifier.geometric_bias.item():.4f}")
                    
        #         # === SCORE RECALCULATION TEST ===
        #         # Now recalculate the score from distance and compare with actual prediction score
        #         pred_logit = final_logits[pred_label].item()
        #         pred_distance = model_distances[pred_label]
                
        #         # Recalculate what the score should be based on distance
        #         # Logit = -distance * temperature_scale + bias
        #         expected_logit = -pred_distance
        #         if target_classifier.use_temperature and target_classifier.logit_scale is not None:
        #             temp_scale = target_classifier.logit_scale.exp().clamp(max=100).item()
        #             expected_logit = expected_logit * temp_scale
        #         if target_classifier.geometric_bias is not None:
        #             expected_logit = expected_logit + target_classifier.geometric_bias.item()
                    
        #         expected_score = torch.sigmoid(torch.tensor(expected_logit)).item()
        #         actual_score = score
                
        #         print(f"    Score Analysis:")
        #         print(f"      Actual prediction score: {actual_score:.6f}")
        #         print(f"      Expected score from distance: {expected_score:.6f}")
        #         print(f"      Score difference: {abs(actual_score - expected_score):.6f}")
        #         print(f"      Pred logit (from model): {pred_logit:.6f}")
        #         print(f"      Expected logit (from distance): {expected_logit:.6f}")
        #         print(f"      Logit difference: {abs(pred_logit - expected_logit):.6f}")
                
        #         if target_classifier.use_temperature and target_classifier.logit_scale is not None:
        #             print(f"      Temperature scale: {temp_scale:.6f}")
                    
        #         # === INFERENCE CONSISTENCY CHECK ===
        #         # Check if our calculations match the actual inference logic from _predict_by_feat_single
        #         # The model uses cls_score.max(dim=-1) to get the highest logit, then sigmoid
        #         max_logit_value, max_logit_class = final_logits.max(dim=-1)
        #         inference_score = max_logit_value.sigmoid().item()
        #         inference_class = max_logit_class.item()
                
        #         print(f"    Inference Logic Check:")
        #         print(f"      Model's max logit class: {inference_class} -> {labels[inference_class] if inference_class < len(labels) else 'INVALID'}")
        #         print(f"      Model's inference score: {inference_score:.6f}")
        #         print(f"      Matches prediction class: {inference_class == pred_label}")
        #         print(f"      Matches prediction score: {abs(inference_score - actual_score) < 1e-6}")
                
        #         # Show top 3 predictions for debugging
        #         top3_logits = torch.topk(final_logits, 3)
        #         top3_indices = top3_logits.indices.cpu().numpy()
        #         top3_values = top3_logits.values.cpu().numpy()
        #         print(f"    Top 3 logits: {top3_indices} -> {[labels[i] if i < len(labels) else 'INVALID' for i in top3_indices]}")
        #         print(f"    Top 3 values: {top3_values}")
                
        #         # Show model distances to top predictions
        #         print(f"    Model top 3 closest: {model_nn_idx} and others -> {[labels[i] if i < len(labels) else 'INVALID' for i in np.argsort(model_distances)[:3]]}")
        #         print(f"    Model top 3 distances: {np.sort(model_distances)[:3]}")
        #     print("=" * 60)

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


def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def overlay_detection_thumbnails(ax: Axes, examples: List[Dict]):
    """Overlay detection thumbnails with prediction labels on the UMAP plot."""
    
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
    
    # Group examples by approximate regions to avoid overlapping
    grid_size = int(np.ceil(np.sqrt(len(examples))))
    region_width = data_width / grid_size
    region_height = data_height / grid_size
    
    # Adjust positions to avoid overlaps - create a grid of possible positions
    positions_used = {}  # To track used positions
    
    for example in examples:
        x, y = example['umap_coords']
        crop = example['crop_image']
        fallback_level = example['fallback_level']
        pred_node = example['pred_node']
        gt_leaf = example['gt_leaf']
        
        # Calculate grid cell for this point
        grid_x = min(int((x - xlim[0]) / region_width), grid_size-1)
        grid_y = min(int((y - ylim[0]) / region_height), grid_size-1)
        grid_key = (grid_x, grid_y)
        
        # If position is already used, slightly offset
        offset_factor = 0
        original_grid_key = grid_key
        while grid_key in positions_used:
            offset_factor += 1
            angle = offset_factor * 45  # Try 8 directions
            distance = min(region_width, region_height) * 0.3 * (offset_factor // 8 + 1)
            offset_x = distance * np.cos(np.radians(angle))
            offset_y = distance * np.sin(np.radians(angle))
            
            # Try new position
            new_x = x + offset_x
            new_y = y + offset_y
            
            # Ensure we stay within bounds
            if (new_x < xlim[0] or new_x > xlim[1] or 
                new_y < ylim[0] or new_y > ylim[1]):
                continue
                
            grid_x = min(int((new_x - xlim[0]) / region_width), grid_size-1)
            grid_y = min(int((new_y - ylim[0]) / region_height), grid_size-1)
            grid_key = (grid_x, grid_y)
            
            # Prevent infinite loop
            if offset_factor > 20:
                grid_key = original_grid_key  # Revert to original if can't find space
                break
        
        # Mark position as used
        positions_used[grid_key] = True
        
        # Adjusted display coordinates - center in the grid cell with slight jitter
        center_x = xlim[0] + (grid_x + 0.5) * region_width
        center_y = ylim[0] + (grid_y + 0.5) * region_height
        jitter_x = region_width * 0.2 * np.random.uniform(-1, 1)
        jitter_y = region_height * 0.2 * np.random.uniform(-1, 1)
        display_x = center_x + jitter_x
        display_y = center_y + jitter_y
        
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


# -----------------------------------------------------------------------------
# Main Plotting Function
# -----------------------------------------------------------------------------

def plot_combined_umap_with_detections(model_path: str, config_path: str,
                                      save_path: Optional[str] = None,
                                      num_examples: int = 20,  # Fixed to match argument parser default
                                      random_state: int = 42,
                                      min_score: float = 0.3,
                                      iou_threshold: float = 0.5,
                                      use_mds: bool = False):
    """
    Create the combined UMAP/MDS figure with prototype embeddings and detection examples.
    Uses forward hooks to capture actual query embeddings for scientific accuracy.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model config file (annotation file will be inferred from this)
        save_path: Optional path to save the figure
        num_examples: Number of detection examples to overlay (default: 12)
        random_state: Random seed for UMAP/MDS
        min_score: Minimum confidence score for detections
        iou_threshold: IoU threshold for matching detected bboxes to ground truth (default: 0.5)
        use_mds: If True, use MetricMDS instead of UMAP for better distance preservation
    """
    
    print("Setting up dataset and model for hooks-based embedding extraction...")
    # Load config and dataset first
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # Initialize model for inference with hooks
    model = init_detector(config_path, model_path, device='cuda')
    
    print("Loading prototype embeddings and hierarchy (without UMAP fitting)...")
    # Load prototype embeddings and basic data structures first, but don't fit UMAP yet
    try:
        # Use the model's prototypes property to get embeddings in the same space
        # as the query embeddings captured from get_projected_features hook
        last_branch_idx = get_last_classification_branch_index(model) - 1
        print(f"Using embeddings from branch {last_branch_idx}: bbox_head.cls_branches.{last_branch_idx}.prototypes")
        target_classifier = getattr(model.bbox_head.cls_branches, str(last_branch_idx))
        prototype_embeddings = target_classifier.prototypes.detach().cpu().numpy()

        # Extract basic hierarchy data from config for initial setup
        from mmengine.fileio import load
        cfg = Config.fromfile(config_path)
        
        # Extract annotation file path
        ann_file_path = None
        if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
            ann_file = cfg.test_dataloader.dataset.ann_file
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg.test_dataloader.dataset, 'data_root') else ''
            if data_root and not os.path.isabs(ann_file):
                ann_file_path = os.path.join(data_root, ann_file)
            else:
                ann_file_path = ann_file
        
        if ann_file_path is None:
            data_root = cfg.test_dataloader.dataset.data_root if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset') and hasattr(cfg.test_dataloader.dataset, 'data_root') else 'data/aircraft'
            ann_file_path = os.path.join(data_root, 'aircraft_test.json')
        
        # Load annotation data for hierarchy
        ann = load(ann_file_path)
        if not isinstance(ann, dict) or "categories" not in ann or "taxonomy" not in ann:
            print("Error: Annotation file is missing 'categories' or 'taxonomy' keys.")
            return
        
        categories = ann["categories"]
        labels = [cat["name"] for cat in categories]
        hierarchy = HierarchyTree(ann["taxonomy"])
        hue_map = build_hue_map(hierarchy.root)
        
    except Exception as e:
        print(f"Error loading basic data: {e}")
        return
    
    print("Extracting detection examples with actual query embeddings using hooks...")
    # Extract detection examples using hooks to get ACTUAL query embeddings
    detection_examples = extract_detection_examples_with_hooks(
        model, dataset, prototype_embeddings, hierarchy, labels, num_examples, min_score, iou_threshold)
    
    # Collect query embeddings from examples for joint UMAP fitting
    query_embeddings = None
    if detection_examples:
        query_embeddings = np.array([ex['feature_vector'] for ex in detection_examples])
        print(f"Collected {len(query_embeddings)} ACTUAL query embeddings for joint UMAP fitting")
    
    print("Performing single joint projection fitting with prototypes + query embeddings...")
    # Now do a SINGLE projection fitting with both prototype and query embeddings together
    projection_method = "MetricMDS" if use_mds else "UMAP"
    print(f"Using {projection_method} for projection...")
    embeddings, _projection_model, hierarchy, hue_map, labels, node_coords, query_projections = load_embeddings_and_umap(
        model_path, config_path, random_state, query_embeddings, prototype_embeddings, use_mds, detection_examples)
    
    if embeddings is None or hierarchy is None or labels is None or node_coords is None or hue_map is None:
        print("Failed to load embeddings or related data structures. Exiting.")
        return
    
    print("Using joint UMAP projections for detection examples...")
    # Use the query projections computed during joint UMAP fitting
    if detection_examples and query_projections is not None:
        print(f"Using {len(query_projections)} pre-computed query projections from joint UMAP fitting")
        # Update examples with the joint UMAP coordinates
        for i, example in enumerate(detection_examples):
            if i < len(query_projections):
                example['umap_coords'] = query_projections[i]
                print(f"Example {i}: {example['gt_leaf']} → {example['pred_node']} (level {example['fallback_level']})")
                print(f"  Positioned at UMAP coords: ({query_projections[i][0]:.2f}, {query_projections[i][1]:.2f})")
            else:
                print(f"Warning: Missing projection for example {i}")
                
        # Check if the projections make sense (look for outliers)
        if len(query_projections) > 2:
            distances = []
            for i in range(len(query_projections)):
                # Find closest prototype
                min_dist = float('inf')
                for label, coord in node_coords.items():
                    dist = np.sqrt(np.sum((query_projections[i] - coord)**2))
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
            
            avg_dist = np.mean(distances)
            max_dist = np.max(distances)
            print(f"Average distance to nearest prototype: {avg_dist:.2f}")
            print(f"Maximum distance to nearest prototype: {max_dist:.2f}")
            if max_dist > 3 * avg_dist:
                print("Warning: Some examples are very far from prototypes. Check embedding extraction.")
    
    print("Calculating visual attributes...")
    # Calculate visual attributes for prototypes
    plotted_nodes = set(node_coords.keys())
    depth_cmap = plt.get_cmap('cividis_r')
    
    filtered_labels, filtered_coords, marker_sizes, fill_colors, edge_colors, _, _ = calculate_visual_attributes(
        hierarchy, plotted_nodes, node_coords, labels, hue_map, depth_cmap)
    
    if len(filtered_labels) == 0:
        print("No valid prototype data to plot. Exiting.")
        return
    
    print("Creating plot...")
    # Create the plot with publication-quality dimensions
    fig, ax = plt.subplots(figsize=(20, 18), dpi=120)  # Higher DPI for better detail
    
    # Use a clean, professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set up background with subtle gradient for professional look
    ax.set_facecolor('#f9f9f9')  # Light gray background
    fig.patch.set_facecolor('white')
    
    # Plot taxonomy components with adjusted z-order for better layering
    plot_convex_hulls(ax, plotted_nodes, hierarchy, node_coords, hue_map)
    plot_taxonomy_skeleton(ax, plotted_nodes, hierarchy, node_coords)
    plot_prototype_scatter(ax, filtered_coords, marker_sizes, fill_colors, edge_colors, filtered_labels)
    
    # Overlay detection examples
    if detection_examples:
        overlay_detection_thumbnails(ax, detection_examples)
    
    # Extract model name for title
    model_name = pathlib.Path(model_path).stem.replace('_', ' ').title()
    if len(model_name) > 30:  # Shorten very long model names
        model_name = model_name[:27] + "..."
        
    # Configure plot with publication-quality styling
    projection_name = "MetricMDS" if use_mds else "UMAP"
    ax.set_title(f"{projection_name} Visualization of Prototype Embeddings with Detection Examples\n"
                f"GT-Prediction Relationship Analysis | {model_name}", 
                fontsize=16, pad=20, fontweight='bold')
    
    # Use professional font for axes labels
    ax.set_xlabel(f"{projection_name} Dimension 1", fontsize=14, labelpad=15)
    ax.set_ylabel(f"{projection_name} Dimension 2", fontsize=14, labelpad=15)
    
    # Ensure aspect ratio is maintained for better visual perception
    ax.set_aspect('equal', adjustable='box')
    
    # Remove ticks for cleaner look
    ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                  left=False, right=False, labelbottom=False, labelleft=False)
    
    # Add subtle grid for spatial reference
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # Add legend explaining the colors
    add_detection_border_legend(ax)
    
    # Add professional border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#888888')
        spine.set_linewidth(0.8)
    
    # Adjust layout for better spacing
    fig.tight_layout(pad=2.5)
    
    # Save or show
    if save_path:
        # Add timestamp and parameters to filename for tracking
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name_clean = os.path.basename(model_path).replace('.', '_').replace(' ', '_')
        base, ext = os.path.splitext(save_path)
        detailed_path = f"{base}_{model_name_clean}_{num_examples}ex_{timestamp}{ext}"
        
        # Ensure the directory exists
        save_dir = os.path.dirname(detailed_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate actual number of examples shown
        num_actual_examples = len(detection_examples) if detection_examples else 0
        
        # Save with high resolution and metadata
        print(f"Saving figure to {detailed_path}")
        metadata = {
            'Title': f'UMAP visualization with {num_actual_examples} detection examples',
            'Author': 'Hierarchical Object Detection',
            'Description': f'Model: {model_name}, Examples: {num_actual_examples}, Date: {timestamp}',
            'Keywords': 'UMAP, embeddings, object detection, hierarchy'
        }
        fig.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   pad_inches=0.3, metadata=metadata)
        print(f"Figure saved with {num_actual_examples} detection examples")
    
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
    parser.add_argument('--num-examples', type=int, default=20, 
                       help='Number of detection examples to overlay (default: 20)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for UMAP (default: 42)')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum confidence score for detection examples (default: 0.3)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching detected bboxes to ground truth (default: 0.5)')
    parser.add_argument('--use-mds', action='store_true',
                       help='Use MetricMDS instead of UMAP for better distance preservation (slower but more accurate)')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, args.save_name)
    
    plot_combined_umap_with_detections(
        model_path=args.model_path,
        config_path=args.config,
        save_path=save_path,
        num_examples=args.num_examples,
        random_state=args.random_state,
        min_score=args.min_score,
        iou_threshold=args.iou_threshold,
        use_mds=args.use_mds
    )


if __name__ == '__main__':
    main()
