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
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Set
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
        """Register forward hooks on EmbeddingClassifier modules in the last classification branch"""
        hooks_registered = 0
        
        # Determine the last classification branch index  
        last_branch_idx = get_last_classification_branch_index(model)
        print(f"Model reports last branch index as: {last_branch_idx}")
        
        # TEST: Try cls_branches.0 instead - you suspect branches might be stored inversely
        last_branch_idx = 0
        target_branch = f'cls_branches.{last_branch_idx}'
        print(f"Testing with classification branch: {target_branch} (user theory: branches stored inversely)")
        
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
                            
                            # Only keep embeddings from the decoder stage (300 queries)
                            # The encoder stage produces much larger embedding counts (~19k)
                            if embeddings_batch.shape[-2] <= 1000:  # Reasonable threshold for decoder queries
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
        
    def get_embeddings_for_detections(self, pred_instances, num_detections):
        """
        Extract embeddings corresponding to the detections using bbox_index from pred_instances.
        
        Args:
            pred_instances: The prediction results containing bbox_index
            num_detections: Number of final detections (for validation)
            
        Returns:
            Selected embeddings corresponding to final detections, or None if no embeddings
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
        
        # Use bbox_index from pred_instances (this is the scientifically accurate approach)
        if hasattr(pred_instances, 'bbox_index'):
            bbox_index = pred_instances.bbox_index
            # print(f"Using bbox_index from pred_instances: {len(all_embeddings)} embeddings -> {len(bbox_index)} selected indices")
            
            # Convert tensor to numpy if needed
            if hasattr(bbox_index, 'cpu'):
                bbox_index = bbox_index.cpu().numpy()
            
            # Validate that we have enough embeddings
            if len(all_embeddings) > max(bbox_index):
                selected_embeddings = all_embeddings[bbox_index]
                # print(f"Successfully mapped {len(selected_embeddings)} embeddings using bbox_index")
                return selected_embeddings
            else:
                print(f"Warning: bbox_index max ({max(bbox_index)}) >= embedding count ({len(all_embeddings)})")
        
        # Fallback to previous logic if bbox_index not available
        print(f"Falling back to truncation: bbox_index not available from pred_instances")
        if len(all_embeddings) >= num_detections:
            return all_embeddings[:num_detections]
        else:
            print(f"Warning: Only {len(all_embeddings)} embeddings for {num_detections} detections")
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
                    
                    # Get corresponding query embeddings using bbox_index from pred_instances
                    query_embeddings = collector.get_embeddings_for_detections(pred_instances, num_detections)
                    
                    if query_embeddings is not None:
                        # Add embeddings to the result
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

def load_embeddings_and_umap(model_path: str, ann_file_path: str, random_state: int = 42, query_embeddings: Optional[np.ndarray] = None, prototype_embeddings: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[umap.UMAP], Optional[HierarchyTree], Optional[Dict[str, float]], Optional[List[str]], Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load embeddings, fit UMAP, and prepare data structures.
    
    Args:
        model_path: Path to model checkpoint
        ann_file_path: Path to annotation file
        random_state: Random seed for UMAP
        query_embeddings: Optional detection query embeddings to include in UMAP fitting
        prototype_embeddings: Optional pre-computed prototype embeddings (if None, will load model and extract prototypes)
        
    Returns:
        Tuple of (embeddings, umap_model, hierarchy, hue_map, labels, node_coords, query_projections)
        query_projections will be None if no query_embeddings were provided
        
    Note:
        Both prototype and query embeddings use the same transformation pipeline via the
        EmbeddingClassifier's 'prototypes' property, ensuring they are in the same space.
    """
    try:
        model_path_obj = pathlib.Path(model_path)
        ann_file_path_obj = pathlib.Path(ann_file_path)

        if not model_path_obj.exists():
            print(f"Error: Model file not found at '{model_path_obj}'")
            return None, None, None, None, None, None, None
        if not ann_file_path_obj.exists():
            print(f"Error: Annotation file not found at '{ann_file_path_obj}'")
            return None, None, None, None, None, None, None

        # Use pre-computed embeddings if provided, otherwise load model and extract prototypes
        if prototype_embeddings is not None:
            embeddings = prototype_embeddings
        else:
            # Load the model to access the prototypes property for properly transformed embeddings
            print(f"Loading model from {model_path}")
            
            # Determine config path from model path
            config_path = str(model_path_obj).replace('.pth', '.py')
            if not pathlib.Path(config_path).exists():
                print(f"Warning: Config file not found at {config_path}, trying to infer...")
                # Try to find config in the same directory
                model_dir = model_path_obj.parent
                config_files = list(model_dir.glob("*.py"))
                if config_files:
                    config_path = str(config_files[0])
                    print(f"Using config file: {config_path}")
                else:
                    raise FileNotFoundError(f"Could not find config file for model at {model_path}")
            
            # Load model
            model = init_detector(config_path, model_path, device='cpu')
            
            # Use the model's prototypes property to get properly transformed embeddings
            # This ensures consistency with query embeddings captured from get_projected_features hook
            # IMPORTANT: Use cls_branches.0 to match the query embedding collection approach
            prototype_branch_idx = 0  # Use same branch as query embeddings (branches stored in reverse)
            target_classifier = getattr(model.bbox_head.cls_branches, str(prototype_branch_idx))
            embeddings = target_classifier.prototypes.detach().cpu().numpy()
            print(f"Using model's prototypes property from branch {prototype_branch_idx} (matching query embedding branch)")
            print(f"Extracted {len(embeddings)} prototype embeddings with shape {embeddings.shape}")

        # Load annotation data
        ann = load(ann_file_path_obj)
        if not isinstance(ann, dict) or "categories" not in ann or "taxonomy" not in ann:
            print("Error: Annotation file is missing 'categories' or 'taxonomy' keys.")
            return None, None, None, None, None, None, None
        
        categories = ann["categories"]
        if len(embeddings) != len(categories):
            print(f"Warning: Number of embeddings ({len(embeddings)}) != categories ({len(categories)})")        # STRATEGY: Fit UMAP on prototypes only, then transform queries
        # This approach ensures the prototype layout remains stable and scientifically meaningful,
        # while query embeddings are positioned relative to the learned prototype structure
        
        # Optimize UMAP parameters for 256-dimensional embeddings to reduce volatility
        n_neighbors = min(30, len(embeddings)-1 if len(embeddings) > 1 else 1)  # Increased for stability
        min_dist = 0.1
        
        # Enhanced UMAP parameters for better stability with high-dimensional data
        reducer = umap.UMAP(
            random_state=random_state, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            metric='euclidean',  # Match model's classification approach
            spread=1.0,
            # Key stability improvements for 256D embeddings:
            n_epochs=500,  # More training epochs for convergence
            learning_rate=1.0,  # Default learning rate
            init='spectral',  # More stable than 'random' for high-dim data
            low_memory=False,  # Use more memory for better quality
            # Preprocessing to reduce dimensionality before UMAP
            densmap=False,  # Keep False for consistency
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            # Transform quality
            transform_seed=random_state,
            transform_mode='embedding'  # Better for transform() quality
        )
        
        print(f"Fitting UMAP on {len(embeddings)} prototype embeddings...")
        umap_2d = reducer.fit_transform(embeddings)
        print(f"UMAP fitting completed. Prototype shape: {umap_2d.shape}")

        # Ensure we have a proper numpy array
        if hasattr(umap_2d, 'toarray'):  # For sparse matrix
            umap_2d = umap_2d.toarray()
        else:  # Already dense
            umap_2d = np.asarray(umap_2d)
        
        # Transform query embeddings using the fitted UMAP model
        query_projections = None
        if query_embeddings is not None and len(query_embeddings) > 0:
            # Verify query embeddings are the same dimensionality as prototypes
            if query_embeddings.shape[1] != embeddings.shape[1]:
                print(f"Warning: Query embedding dimensionality ({query_embeddings.shape[1]}) " +
                      f"doesn't match prototype dimensionality ({embeddings.shape[1]}). Skipping query transform.")
            else:
                # Print some stats to verify the embeddings look reasonable
                proto_norm = np.mean(np.linalg.norm(embeddings, axis=1))
                query_norm = np.mean(np.linalg.norm(query_embeddings, axis=1))
                print(f"Prototype embeddings mean norm: {proto_norm:.4f}")
                print(f"Query embeddings mean norm: {query_norm:.4f}")
                
                # Apply the same preprocessing to query embeddings if it was used for prototypes
                preprocessed_query_embeddings = query_embeddings
                if hasattr(reducer, '_preprocessing_applied') and reducer._preprocessing_applied:
                    pca_preprocessor = getattr(reducer, '_pca_preprocessor')
                    preprocessed_query_embeddings = pca_preprocessor.transform(query_embeddings)
                    print(f"Applied same PCA preprocessing to queries: {query_embeddings.shape[1]}D → {preprocessed_query_embeddings.shape[1]}D")
                
                print(f"Transforming {len(query_embeddings)} query embeddings using fitted UMAP...")
                try:
                    query_projections = reducer.transform(preprocessed_query_embeddings)
                    print(f"Query transformation completed. Shape: {query_projections.shape}")
                    
                    # Verify the query projections are reasonable
                    proto_x_range = (np.min(umap_2d[:, 0]), np.max(umap_2d[:, 0]))
                    proto_y_range = (np.min(umap_2d[:, 1]), np.max(umap_2d[:, 1]))
                    query_x_range = (np.min(query_projections[:, 0]), np.max(query_projections[:, 0]))
                    query_y_range = (np.min(query_projections[:, 1]), np.max(query_projections[:, 1]))
                    
                    print(f"Prototype UMAP range: X={proto_x_range}, Y={proto_y_range}")
                    print(f"Query UMAP range: X={query_x_range}, Y={query_y_range}")
                    
                except Exception as e:
                    print(f"Warning: Query transformation failed: {e}")
                    print("This can happen with UMAP transform() on distant embeddings. Using prototype-only visualization.")
                    query_projections = None
                
                # Calculate overlap metrics
                x_overlap = max(0, min(proto_x_range[1], query_x_range[1]) - max(proto_x_range[0], query_x_range[0]))
                y_overlap = max(0, min(proto_y_range[1], query_y_range[1]) - max(proto_y_range[0], query_y_range[0]))
                proto_x_span = proto_x_range[1] - proto_x_range[0]
                proto_y_span = proto_y_range[1] - proto_y_range[0]
                x_overlap_ratio = x_overlap / proto_x_span if proto_x_span > 0 else 0
                y_overlap_ratio = y_overlap / proto_y_span if proto_y_span > 0 else 0
                print(f"Spatial overlap: X={x_overlap_ratio:.2%}, Y={y_overlap_ratio:.2%}")
        else:
            print("No query embeddings provided - using prototype-only visualization")

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
                    'pred_node': pred_node,
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
    
    # Random sampling for final selection to ensure diversity
    if len(examples) > num_examples:
        print(f"Randomly sampling {num_examples} examples from {len(examples)} collected examples")
        random.shuffle(examples)  # Shuffle for random selection
        selected_examples = examples[:num_examples]
    else:
        selected_examples = examples
    
    # Show the final selection breakdown after random sampling
    if len(examples) > num_examples:
        final_selected_counts = [0] * 6
        for ex in selected_examples:
            final_selected_counts[ex['fallback_level']] += 1
        
        print(f"After random sampling:")
        for level in range(6):
            if final_selected_counts[level] > 0:
                print(f"  {fallback_names[level]}: {final_selected_counts[level]} examples")
    
    return selected_examples


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


def extract_detection_examples_for_embeddings(dataset, results, embeddings, 
                                              hierarchy: HierarchyTree, labels: List[str],
                                              num_examples: int = 12, min_score: float = 0.3) -> List[Dict]:
    """
    Extract detection examples with their feature vectors for joint UMAP fitting.
    This version carefully extracts accurate embeddings for better UMAP projection.
    """
    examples_by_level = {level: [] for level in range(6)}
    
    print(f"Extracting {num_examples} detection examples for embedding collection...")
    
    # Check if we can get embeddings from results
    has_embeddings = False
    for result in results[:min(10, len(results))]:
        if 'pred_instances' in result and 'embeddings' in result['pred_instances']:
            has_embeddings = True
            print("✅ Detection results contain embeddings - will use actual embedding vectors.")
            break
    
    if not has_embeddings:
        print("⚠️ Detection results don't contain embeddings - will use prototype embeddings as approximation.")
        print("   This may result in less accurate positioning in the UMAP visualization.")
    
    # Collect examples from results - search more images for better diversity
    for img_idx, result in enumerate(results[:min(300, len(results))]):
        # Get image info and ground truth
        img_info = dataset.get_data_info(img_idx)
        img_path = img_info['img_path']
        gt_instances = img_info['instances']
        
        # Get predictions
        pred_instances = result['pred_instances']
        pred_bboxes = pred_instances['bboxes'].numpy()
        pred_scores = pred_instances['scores'].numpy()
        pred_labels = pred_instances['labels'].numpy()
        
        # Extract embeddings if available (more accurate than using prototype embeddings)
        pred_embeddings = None
        if 'embeddings' in pred_instances:
            pred_embeddings = pred_instances['embeddings'].numpy()
        
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
            
            # Get feature vector - use detection embedding if available, otherwise prototype
            if pred_embeddings is not None and i < len(pred_embeddings):
                feature_vector = pred_embeddings[i]
            elif pred_label < len(embeddings):
                feature_vector = embeddings[pred_label]
            else:
                continue
                
            example = {
                'crop_image': crop_resized,
                'feature_vector': feature_vector,
                'umap_coords': None,  # Will be filled later
                'gt_leaf': gt_leaf,
                'pred_node': pred_node,
                'fallback_level': fallback_level,
                'confidence': score
            }
            
            examples_by_level[fallback_level].append(example)
            print(f"  Found Level {fallback_level}: GT={gt_leaf}, Pred={pred_node}, Score={score:.3f}")
    
    # Balanced sampling across fallback levels
    examples = []
    examples_per_level = max(1, num_examples // 6)
    fallback_names = ['Leaf Correct', 'Parent', 'Grandparent', 'Sibling', 'Cousin', 'Off-branch']
    
    for level in range(6):
        available = examples_by_level[level]
        if available:
            # Sort by confidence score to get the best examples
            sorted_examples = sorted(available, key=lambda x: x['confidence'], reverse=True)
            selected = sorted_examples[:examples_per_level]
            examples.extend(selected)
            print(f"  Selected {len(selected)} examples from {fallback_names[level]} (Level {level})")
    
    # Fill remaining slots
    while len(examples) < num_examples:
        added = False
        for level in range(6):
            if len(examples) >= num_examples:
                break
            available = examples_by_level[level]
            already_taken = sum(1 for ex in examples if ex['fallback_level'] == level)
            if len(available) > already_taken:
                examples.append(available[already_taken])
                added = True
        if not added:
            break
    
    # Random sampling if we have more examples than needed
    if len(examples) > num_examples:
        print(f"Randomly sampling {num_examples} examples from {len(examples)} collected examples for embedding collection")
        random.shuffle(examples)
        examples = examples[:num_examples]
    
    print(f"Extracted {len(examples)} examples total for embedding collection")
    return examples


def extract_detection_examples(dataset, results, embeddings, 
                              hierarchy: HierarchyTree, labels: List[str],
                              num_examples: int = 12, min_score: float = 0.3) -> List[Dict]:
    """
    Extract K detection examples with balanced sampling across fallback levels.
    
    Returns:
        List of dicts with keys: 'crop_image', 'feature_vector', 'umap_coords', 
        'gt_leaf', 'pred_node', 'fallback_level', 'confidence'
    """
    # Organize examples by fallback level for balanced sampling
    examples_by_level = {level: [] for level in range(6)}  # 0-5 fallback levels
    
    print(f"Extracting {num_examples} detection examples with balanced fallback sampling...")
    
    # Collect examples from results, organizing by fallback level
    for img_idx, result in enumerate(results[:min(300, len(results))]):  # Search more images for diversity
            
        # Get image info and ground truth
        img_info = dataset.get_data_info(img_idx)
        img_path = img_info['img_path']
        gt_instances = img_info['instances']
        
        # Get predictions
        pred_instances = result['pred_instances']
        pred_bboxes = pred_instances['bboxes'].numpy()
        pred_scores = pred_instances['scores'].numpy()
        pred_labels = pred_instances['labels'].numpy()
        
        # Extract embeddings if available
        pred_embeddings = None
        if 'embeddings' in pred_instances:
            pred_embeddings = pred_instances['embeddings'].numpy()
        
        # Load image
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
                # IoU check for matching
                if bbox_iou(bbox, gt_bbox) > iou_threshold:
                    gt_leaf = labels[gt_inst['bbox_label']]
                    break
            
            if gt_leaf is None:
                continue  # Skip if no ground truth match
                
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
            
            # Resize crop to standard size
            crop_size = 96
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            
            # Get feature vector - use detection embedding if available, otherwise prototype
            if pred_embeddings is not None and i < len(pred_embeddings):
                # Use the actual embedding from the detection, which is more accurate
                feature_vector = pred_embeddings[i].copy()  # Make a copy to avoid modifying original
                print(f"Using actual detection embedding for {pred_node} (norm: {np.linalg.norm(feature_vector):.3f})")
            elif pred_label < len(embeddings):
                # Use the prototype as an approximation - less accurate but still works
                feature_vector = embeddings[pred_label].copy()
                print(f"Using prototype embedding for {pred_node} as fallback")
            else:
                continue
                
            example = {
                'crop_image': crop_resized,
                'feature_vector': feature_vector,
                'umap_coords': None,  # Will be filled later
                'gt_leaf': gt_leaf,
                'pred_node': pred_node,
                'fallback_level': fallback_level,
                'confidence': score
            }
            
            examples_by_level[fallback_level].append(example)
            print(f"  Found Level {fallback_level}: GT={gt_leaf}, Pred={pred_node}, Score={score:.3f}")
    
    # Balanced sampling: take examples from each level
    examples = []
    examples_per_level = max(1, num_examples // 6)  # Distribute evenly across 6 levels
    
    fallback_names = ['Exact Match', 'Parent', 'Grandparent', 'Sibling', 'Cousin', 'Off-branch']
    
    for level in range(6):
        available = examples_by_level[level]
        if available:
            # Sort by confidence score before selecting
            sorted_examples = sorted(available, key=lambda x: x['confidence'], reverse=True)
            # Take up to examples_per_level from this fallback level
            selected = sorted_examples[:examples_per_level]
            examples.extend(selected)
            print(f"  Selected {len(selected)} examples from {fallback_names[level]} (Level {level})")
    
    # If we need more examples and have room, fill from any remaining
    while len(examples) < num_examples:
        added = False
        for level in range(6):
            if len(examples) >= num_examples:
                break
            available = examples_by_level[level]
            already_taken = sum(1 for ex in examples if ex['fallback_level'] == level)
            if len(available) > already_taken:
                examples.append(available[already_taken])
                added = True
        if not added:
            break  # No more examples available
    
    # Random sampling if we have more examples than needed
    if len(examples) > num_examples:
        print(f"Randomly sampling {num_examples} examples from {len(examples)} collected examples")
        random.shuffle(examples)
        examples = examples[:num_examples]
    
    print(f"Extracted {len(examples)} examples total with balanced fallback distribution")
    return examples


def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate intersection
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


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
                                      iou_threshold: float = 0.5):
    """
    Create the combined UMAP figure with prototype embeddings and detection examples.
    Uses forward hooks to capture actual query embeddings for scientific accuracy.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to model config file
        save_path: Optional path to save the figure
        num_examples: Number of detection examples to overlay (default: 20)
        random_state: Random seed for UMAP
        min_score: Minimum confidence score for detections
        iou_threshold: IoU threshold for matching detected bboxes to ground truth (default: 0.5)
    """
    
    print("Setting up dataset and model for hooks-based embedding extraction...")
    # Load config and dataset first
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # Extract annotation file path from the dataset config with proper data root handling
    ann_file_relative = cfg.test_dataloader.dataset.ann_file
    data_root = cfg.test_dataloader.dataset.get('data_root', '')
    
    # Construct full annotation file path
    if data_root and not os.path.isabs(ann_file_relative):
        ann_file_path = os.path.join(data_root, ann_file_relative)
    else:
        ann_file_path = ann_file_relative
        
    print(f"Using annotation file: {ann_file_path} (data_root: {data_root}, relative: {ann_file_relative})")
    
    # Initialize model for inference with hooks
    model = init_detector(config_path, model_path, device='cuda')
    
    print("Loading prototype embeddings and hierarchy (without UMAP fitting)...")
    # Load prototype embeddings and basic data structures first, but don't fit UMAP yet
    try:
        ann_file_path_obj = pathlib.Path(ann_file_path)

        # Use the model's prototypes property to get embeddings in the same space
        # as the query embeddings captured from get_projected_features hook
        last_branch_idx = get_last_classification_branch_index(model)
        print(f"Using embeddings from branch {last_branch_idx}: bbox_head.cls_branches.{last_branch_idx}.prototypes")
        target_classifier = getattr(model.bbox_head.cls_branches, str(last_branch_idx))
        prototype_embeddings = target_classifier.prototypes.detach().cpu().numpy()

        # Load annotation data for hierarchy
        ann = load(ann_file_path_obj)
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
    
    print("Performing single joint UMAP fitting with prototypes + query embeddings...")
    # Now do a SINGLE UMAP fitting with both prototype and query embeddings together
    embeddings, _umap_model, hierarchy, hue_map, labels, node_coords, query_projections = load_embeddings_and_umap(
        model_path, ann_file_path, random_state, query_embeddings, prototype_embeddings)
    
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
    ax.set_title(f"UMAP Visualization of Prototype Embeddings with Detection Examples\n"
                f"GT-Prediction Relationship Analysis | {model_name}", 
                fontsize=16, pad=20, fontweight='bold')
    
    # Use professional font for axes labels
    ax.set_xlabel("UMAP Dimension 1", fontsize=14, labelpad=15)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14, labelpad=15)
    
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
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    main()
