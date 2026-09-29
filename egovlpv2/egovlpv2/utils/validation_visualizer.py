# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from torch.utils.data import DataLoader


def visualize_dataset_samples(data_loader, num_samples=20, save_dir=None, dataset_name="dataset"):
    """
    Visualize random samples from dataset to check video-text alignment
    
    Args:
        data_loader: DataLoader object
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualization results
        dataset_name: Name of the dataset for file naming
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get random samples
    total_samples = len(data_loader.dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"Visualizing {len(sample_indices)} samples from {dataset_name}")
    
    for idx, sample_idx in enumerate(sample_indices):
        try:
            # Get sample
            sample = data_loader.dataset[sample_idx]
            video_frames = sample['video']  # [T, C, H, W]
            text = sample['text']
            meta = sample.get('meta', {})
            
            # Create visualization
            fig = create_sample_visualization(video_frames, text, meta, sample_idx)
            
            # Save or show
            if save_dir:
                save_path = os.path.join(save_dir, f"{dataset_name}_sample_{idx+1:03d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization: {save_path}")
            else:
                plt.show()
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error visualizing sample {sample_idx}: {e}")
            continue


def create_sample_visualization(video_frames, text, meta, sample_idx):
    """
    Create a single sample visualization with video frames and text
    
    Args:
        video_frames: Video tensor [T, C, H, W]
        text: Text caption string
        meta: Metadata dictionary
        sample_idx: Sample index
        
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    # Select frames to display (evenly spaced)
    num_frames = video_frames.shape[0]
    display_frames = min(8, num_frames)  # Show at most 8 frames
    frame_indices = np.linspace(0, num_frames-1, display_frames, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Title with sample info
    sample_info = f"Sample {sample_idx}"
    if 'clip_uid' in meta:
        sample_info += f" | Clip: {meta['clip_uid']}"
    if 'duration_sec' in meta:
        sample_info += f" | Duration: {meta['duration_sec']:.2f}s"
    
    fig.suptitle(sample_info, fontsize=14, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(3, display_frames, height_ratios=[0.3, 2, 0.7], hspace=0.3)
    
    # Add text at the top
    text_ax = fig.add_subplot(gs[0, :])
    text_ax.text(0.5, 0.5, f"Text: {text}", ha='center', va='center', 
                fontsize=12, wrap=True, transform=text_ax.transAxes)
    text_ax.axis('off')
    
    # Display frames
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[1, i])
        
        # Convert tensor to displayable image
        frame = video_frames[frame_idx]  # [C, H, W]
        if frame.shape[0] == 3:  # RGB
            # Convert from [C, H, W] to [H, W, C] and denormalize if needed
            frame_np = frame.permute(1, 2, 0).numpy()
            
            # If normalized, denormalize (assuming ImageNet normalization)
            if frame_np.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame_np = frame_np * std + mean
            
            # Clip to valid range
            frame_np = np.clip(frame_np, 0, 1)
        else:
            # Handle other formats if needed
            frame_np = frame.squeeze().numpy()
        
        ax.imshow(frame_np)
        ax.set_title(f"Frame {frame_idx}/{num_frames-1}", fontsize=10)
        ax.axis('off')
    
    # Add metadata at the bottom
    meta_ax = fig.add_subplot(gs[2, :])
    meta_text = create_metadata_text(meta)
    meta_ax.text(0.5, 0.5, meta_text, ha='center', va='center', 
                fontsize=10, transform=meta_ax.transAxes)
    meta_ax.axis('off')
    
    return fig


def create_metadata_text(meta):
    """Create formatted metadata text for display"""
    meta_lines = []
    
    # Add relevant metadata fields
    if 'dataset' in meta:
        meta_lines.append(f"Dataset: {meta['dataset']}")
    
    if 'paths' in meta:
        paths = meta['paths']
        if isinstance(paths, list) and len(paths) > 0:
            video_path = os.path.basename(paths[0])
            meta_lines.append(f"Video: {video_path}")
    
    if 'action_uid' in meta:
        meta_lines.append(f"Action UID: {meta['action_uid']}")
    
    if 'structured_verb' in meta:
        meta_lines.append(f"Verb: {meta['structured_verb']}")
    
    return " | ".join(meta_lines) if meta_lines else "No metadata available"


def visualize_similarity_results(similarity_matrix, texts, video_paths, top_k=5, save_dir=None):
    """
    Visualize similarity matrix results with top matches
    
    Args:
        similarity_matrix: Similarity matrix [N, N]
        texts: List of text captions
        video_paths: List of video file paths
        top_k: Number of top matches to display
        save_dir: Directory to save results
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    n_samples = similarity_matrix.shape[0]
    
    # Create overall similarity heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix.cpu().numpy(), cmap='viridis', aspect='auto')
    
    ax.set_xlabel('Video Index')
    ax.set_ylabel('Text Index')
    ax.set_title('Video-Text Similarity Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Similarity Score')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'similarity_matrix.png'), dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)
    
    # Create top matches visualization for a few samples
    sample_indices = random.sample(range(n_samples), min(5, n_samples))
    
    for sample_idx in sample_indices:
        create_top_matches_visualization(
            similarity_matrix, texts, video_paths, sample_idx, top_k, save_dir
        )


def create_top_matches_visualization(similarity_matrix, texts, video_paths, sample_idx, top_k, save_dir):
    """Create visualization showing top matches for a specific sample"""
    
    # Get similarities for this text to all videos
    similarities = similarity_matrix[sample_idx, :]
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    fig, axes = plt.subplots(2, top_k, figsize=(3*top_k, 6))
    if top_k == 1:
        axes = axes.reshape(2, 1)
    
    # Current text
    current_text = texts[sample_idx]
    fig.suptitle(f"Top {top_k} Video Matches for Text {sample_idx}\n\"{current_text[:100]}...\"", 
                fontsize=12, fontweight='bold')
    
    for i, video_idx in enumerate(top_indices):
        similarity_score = similarities[video_idx].item()
        is_correct = video_idx == sample_idx
        
        # Load and display video frame (first frame)
        video_path = video_paths[video_idx]
        try:
            # This is a simplified version - in practice you might want to load actual frames
            axes[0, i].text(0.5, 0.5, f"Video {video_idx}\n{os.path.basename(video_path)}", 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            if is_correct:
                axes[0, i].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                                     edgecolor='green', facecolor='none'))
        except:
            axes[0, i].text(0.5, 0.5, f"Video {video_idx}\nUnavailable", 
                           ha='center', va='center', transform=axes[0, i].transAxes)
        
        axes[0, i].set_title(f"Rank {i+1} | Sim: {similarity_score:.3f}")
        axes[0, i].axis('off')
        
        # Display corresponding text
        corresponding_text = texts[video_idx]
        axes[1, i].text(0.5, 0.5, f"\"{corresponding_text[:50]}...\"", 
                       ha='center', va='center', transform=axes[1, i].transAxes, 
                       fontsize=8, wrap=True)
        axes[1, i].axis('off')
        
        if is_correct:
            axes[1, i].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                                 edgecolor='green', facecolor='none'))
    
    if save_dir:
        save_path = os.path.join(save_dir, f'top_matches_text_{sample_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)


def analyze_and_visualize_errors(similarity_matrix, texts, video_paths, save_dir=None):
    """
    Analyze and visualize common error patterns in similarity results
    
    Args:
        similarity_matrix: Similarity matrix [N, N]
        texts: List of text captions
        video_paths: List of video file paths
        save_dir: Directory to save analysis results
    """
    n_samples = similarity_matrix.shape[0]
    
    # Find incorrect top-1 predictions
    top1_predictions = torch.argmax(similarity_matrix, dim=1)
    correct_predictions = torch.arange(n_samples)
    
    incorrect_mask = top1_predictions != correct_predictions
    incorrect_indices = torch.where(incorrect_mask)[0]
    
    print(f"Found {len(incorrect_indices)} incorrect top-1 predictions out of {n_samples}")
    
    if len(incorrect_indices) > 0 and save_dir:
        # Analyze error patterns
        error_analysis = {
            'total_errors': len(incorrect_indices),
            'error_rate': len(incorrect_indices) / n_samples * 100,
            'examples': []
        }
        
        # Sample some errors for detailed analysis
        sample_errors = random.sample(incorrect_indices.tolist(), min(10, len(incorrect_indices)))
        
        for idx in sample_errors:
            predicted_idx = top1_predictions[idx].item()
            similarity_score = similarity_matrix[idx, predicted_idx].item()
            correct_score = similarity_matrix[idx, idx].item()
            
            error_analysis['examples'].append({
                'text_idx': idx,
                'predicted_video_idx': predicted_idx,
                'correct_video_idx': idx,
                'predicted_similarity': similarity_score,
                'correct_similarity': correct_score,
                'text': texts[idx][:100],
                'predicted_video': os.path.basename(video_paths[predicted_idx]),
                'correct_video': os.path.basename(video_paths[idx])
            })
        
        # Save error analysis
        import json
        with open(os.path.join(save_dir, 'error_analysis.json'), 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        print(f"Error analysis saved to {save_dir}/error_analysis.json") 