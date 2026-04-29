#!/usr/bin/env python3
# Copyright 2024-2025 Image Quality Evaluation Tool
"""
Image Quality Evaluation Tool

Compares generated images/videos with reference images/videos and computes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Usage:
    # Evaluate single image
    python evaluate_quality.py --generated image.png --reference baseline.png --output metrics.json
    
    # Evaluate video (compares frame by frame)
    python evaluate_quality.py --generated video.mp4 --reference baseline.mp4 --output metrics.json
    
    # From your generation code
    from evaluate_quality import evaluate_image_quality
    metrics = evaluate_image_quality(generated_path, reference_path, output_path)
"""

import argparse
import json
import os
import re
import sys
import logging
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    FID_BACKBONE_AVAILABLE = True
except ImportError:
    FID_BACKBONE_AVAILABLE = False

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
    FVD_BACKBONE_AVAILABLE = True
except ImportError:
    FVD_BACKBONE_AVAILABLE = False

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class QualityEvaluator:
    """Image/Video Quality Evaluator"""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize evaluator
        
        Args:
            device: Device for LPIPS computation ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize LPIPS model if available
        self.lpips_model = None
        self.lpips_available = LPIPS_AVAILABLE  # Use instance variable
        
        if self.lpips_available:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
                logging.info(f"LPIPS model initialized on {self.device}")
            except Exception as e:
                logging.warning(f"Failed to initialize LPIPS model: {e}")
                self.lpips_available = False
        else:
            logging.warning("LPIPS not available. Install with: pip install lpips")

        self.fid_model = None
        self.fvd_model = None

        if FID_BACKBONE_AVAILABLE:
            try:
                weights = Inception_V3_Weights.IMAGENET1K_V1
                self.fid_model = inception_v3(weights=weights, aux_logits=True)
                self.fid_model.AuxLogits = None
                self.fid_model.fc = nn.Identity()
                self.fid_model.to(self.device)
                self.fid_model.eval()
                logging.info(f"FID backbone initialized on {self.device}")
            except Exception as e:
                logging.warning(f"Failed to initialize FID backbone: {e}")
                self.fid_model = None
        else:
            logging.warning("torchvision inception_v3 unavailable. FID will be skipped.")

        if FVD_BACKBONE_AVAILABLE:
            try:
                weights = R3D_18_Weights.KINETICS400_V1
                self.fvd_model = r3d_18(weights=weights)
                self.fvd_model.fc = nn.Identity()
                self.fvd_model.to(self.device)
                self.fvd_model.eval()
                logging.info(f"FVD backbone initialized on {self.device}")
            except Exception as e:
                logging.warning(f"Failed to initialize FVD backbone: {e}")
                self.fvd_model = None
        else:
            logging.warning("torchvision r3d_18 unavailable. FVD will be skipped.")

    @staticmethod
    def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        sqrt_diag = np.diag(np.sqrt(eigvals + 1e-12))
        return eigvecs @ sqrt_diag @ eigvecs.T

    def _frechet_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if features1.ndim != 2 or features2.ndim != 2:
            raise ValueError("Features should be 2D arrays: [N, D]")
        if features1.shape[1] != features2.shape[1]:
            raise ValueError("Feature dimensions must match for Frechet distance")

        mu1 = np.mean(features1, axis=0)
        mu2 = np.mean(features2, axis=0)

        if features1.shape[0] > 1:
            sigma1 = np.cov(features1, rowvar=False)
        else:
            sigma1 = np.zeros((features1.shape[1], features1.shape[1]), dtype=np.float64)

        if features2.shape[0] > 1:
            sigma2 = np.cov(features2, rowvar=False)
        else:
            sigma2 = np.zeros((features2.shape[1], features2.shape[1]), dtype=np.float64)

        cov_sqrt = self._matrix_sqrt_psd(sigma1 @ sigma2)
        diff = mu1 - mu2
        dist = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * cov_sqrt)
        return float(np.real(dist))

    def _extract_inception_features(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        if self.fid_model is None:
            raise RuntimeError("FID backbone unavailable")

        feats = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        with torch.no_grad():
            total = images.shape[0]
            for i in range(0, total, batch_size):
                batch = images[i:i + batch_size]
                tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(self.device)
                tensor = F.interpolate(tensor, size=(299, 299), mode='bilinear', align_corners=False)
                tensor = (tensor - mean) / std
                output = self.fid_model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
                if hasattr(output, 'logits'):
                    output = output.logits
                feats.append(output.detach().cpu().numpy())

        return np.concatenate(feats, axis=0).astype(np.float64)

    def _extract_r3d_features(self, clips: np.ndarray, batch_size: int = 4) -> np.ndarray:
        if self.fvd_model is None:
            raise RuntimeError("FVD backbone unavailable")

        feats = []
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

        with torch.no_grad():
            total = clips.shape[0]
            for i in range(0, total, batch_size):
                batch = clips[i:i + batch_size]
                tensor = torch.from_numpy(batch).float().permute(0, 4, 1, 2, 3).to(self.device)
                b, c, t, h, w = tensor.shape
                tensor = tensor.reshape(b * t, c, h, w)
                tensor = F.interpolate(tensor, size=(112, 112), mode='bilinear', align_corners=False)
                tensor = tensor.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
                tensor = (tensor - mean) / std
                output = self.fvd_model(tensor)
                feats.append(output.detach().cpu().numpy())

        return np.concatenate(feats, axis=0).astype(np.float64)

    @staticmethod
    def _build_video_clips(frames: list, clip_len: int = 16, max_clips: int = 8) -> np.ndarray:
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError("No frames available to build clips")

        if num_frames < clip_len:
            idx = np.linspace(0, num_frames - 1, num=clip_len).round().astype(int)
            clip = np.stack([frames[i] for i in idx], axis=0)
            return np.expand_dims(clip, axis=0)

        starts = np.linspace(0, num_frames - clip_len, num=min(max_clips, num_frames - clip_len + 1))
        clips = []
        for start in starts.round().astype(int):
            clips.append(np.stack(frames[start:start + clip_len], axis=0))
        return np.stack(clips, axis=0)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of shape (H, W, C) with values in [0, 1]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img) / 255.0
        else:
            raise ValueError(f"Unsupported image format: {image_path}")
        
        return img_array
    
    def preprocess_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess video frame
        
        Args:
            frame: Raw frame from cv2 (BGR format)
            
        Returns:
            numpy array of shape (H, W, C) with values in [0, 1] (RGB format)
        """
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb / 255.0
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute PSNR between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            PSNR value
        """
        return float(psnr(img1, img2, data_range=1.0))
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            SSIM value
        """
        # Convert to grayscale if needed for SSIM computation
        if len(img1.shape) == 3:
            # Use multichannel SSIM
            return float(ssim(img1, img2, data_range=1.0, channel_axis=2))
        else:
            return float(ssim(img1, img2, data_range=1.0))
    
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """
        Compute LPIPS between two images
        
        Args:
            img1, img2: Images as numpy arrays with values in [0, 1]
            
        Returns:
            LPIPS value or None if LPIPS not available
        """
        if not self.lpips_available or self.lpips_model is None:
            return None
        
        try:
            # Convert to tensors and normalize to [-1, 1]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Convert numpy arrays to PIL Images then to tensors
            pil1 = Image.fromarray((img1 * 255).astype(np.uint8))
            pil2 = Image.fromarray((img2 * 255).astype(np.uint8))
            
            tensor1 = transform(pil1).unsqueeze(0).to(self.device)
            tensor2 = transform(pil2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_value = self.lpips_model(tensor1, tensor2)
            
            return float(lpips_value.item())
            
        except Exception as e:
            logging.warning(f"LPIPS computation failed: {e}")
            return None
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize images to match dimensions (resize to smaller one)
        
        Args:
            img1, img2: Images as numpy arrays
            
        Returns:
            Resized images
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize to the smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        if (h1, w1) != (target_h, target_w):
            img1 = cv2.resize(img1, (target_w, target_h))
        
        if (h2, w2) != (target_h, target_w):
            img2 = cv2.resize(img2, (target_w, target_h))
        
        return img1, img2
    
    def evaluate_images(self, generated_path: str, reference_path: str) -> Dict[str, float]:
        """
        Evaluate quality metrics between two images
        
        Args:
            generated_path: Path to generated image
            reference_path: Path to reference image
            
        Returns:
            Dictionary containing metrics
        """
        # Check if reference file exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference file not found: {reference_path}")
        
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Generated file not found: {generated_path}")
        
        # Load and preprocess images
        img_gen = self.preprocess_image(generated_path)
        img_ref = self.preprocess_image(reference_path)
        
        # Resize to match if necessary
        img_gen, img_ref = self.resize_to_match(img_gen, img_ref)
        
        # Compute metrics
        metrics = {}
        
        try:
            metrics['psnr'] = self.compute_psnr(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"PSNR computation failed: {e}")
            metrics['psnr'] = None
        
        try:
            metrics['ssim'] = self.compute_ssim(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"SSIM computation failed: {e}")
            metrics['ssim'] = None
        
        try:
            metrics['lpips'] = self.compute_lpips(img_ref, img_gen)
        except Exception as e:
            logging.warning(f"LPIPS computation failed: {e}")
            metrics['lpips'] = None

        try:
            ref_features = self._extract_inception_features(np.expand_dims(img_ref, axis=0))
            gen_features = self._extract_inception_features(np.expand_dims(img_gen, axis=0))
            metrics['fid'] = self._frechet_distance(ref_features, gen_features)
        except Exception as e:
            logging.warning(f"FID computation failed: {e}")
            metrics['fid'] = None

        metrics['fvd'] = None
        
        return metrics
    
    def evaluate_videos(self, generated_path: str, reference_path: str, 
                       max_frames: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate quality metrics between two videos (frame by frame)
        
        Args:
            generated_path: Path to generated video
            reference_path: Path to reference video
            max_frames: Maximum number of frames to compare (None for all)
            
        Returns:
            Dictionary containing only average metrics (psnr, ssim, lpips)
        """
        # Check if reference file exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference video not found: {reference_path}")
        
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Generated video not found: {generated_path}")
        
        # Open video files
        cap_gen = cv2.VideoCapture(generated_path)
        cap_ref = cv2.VideoCapture(reference_path)
        
        if not cap_gen.isOpened() or not cap_ref.isOpened():
            raise ValueError("Failed to open video files")
        
        psnr_values = []
        ssim_values = []
        lpips_values = []
        gen_frames = []
        ref_frames = []
        frame_idx = 0
        
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                
                ret_gen, frame_gen = cap_gen.read()
                ret_ref, frame_ref = cap_ref.read()
                
                if not ret_gen or not ret_ref:
                    break
                
                # Preprocess frames
                img_gen = self.preprocess_video_frame(frame_gen)
                img_ref = self.preprocess_video_frame(frame_ref)
                
                # Resize to match if necessary
                img_gen, img_ref = self.resize_to_match(img_gen, img_ref)

                gen_frames.append(img_gen)
                ref_frames.append(img_ref)
                
                # Compute metrics for this frame
                try:
                    psnr_val = self.compute_psnr(img_ref, img_gen)
                    psnr_values.append(psnr_val)
                except:
                    pass
                
                try:
                    ssim_val = self.compute_ssim(img_ref, img_gen)
                    ssim_values.append(ssim_val)
                except:
                    pass
                
                try:
                    lpips_val = self.compute_lpips(img_ref, img_gen)
                    if lpips_val is not None:
                        lpips_values.append(lpips_val)
                except:
                    pass
                
                frame_idx += 1
                
                if frame_idx % 30 == 0:  # Progress logging
                    logging.info(f"Processed {frame_idx} frames...")
        
        finally:
            cap_gen.release()
            cap_ref.release()
        
        if frame_idx == 0:
            raise ValueError("No frames could be processed")
        
        # Compute average metrics - only return the 3 average values
        metrics = {}
        
        if psnr_values:
            metrics['psnr'] = float(np.mean(psnr_values))
        else:
            metrics['psnr'] = None
            
        if ssim_values:
            metrics['ssim'] = float(np.mean(ssim_values))
        else:
            metrics['ssim'] = None
            
        if lpips_values:
            metrics['lpips'] = float(np.mean(lpips_values))
        else:
            metrics['lpips'] = None

        try:
            ref_stack = np.stack(ref_frames, axis=0)
            gen_stack = np.stack(gen_frames, axis=0)
            ref_features = self._extract_inception_features(ref_stack)
            gen_features = self._extract_inception_features(gen_stack)
            metrics['fid'] = self._frechet_distance(ref_features, gen_features)
        except Exception as e:
            logging.warning(f"Video FID computation failed: {e}")
            metrics['fid'] = None

        try:
            ref_clips = self._build_video_clips(ref_frames)
            gen_clips = self._build_video_clips(gen_frames)
            ref_features = self._extract_r3d_features(ref_clips)
            gen_features = self._extract_r3d_features(gen_clips)
            metrics['fvd'] = self._frechet_distance(ref_features, gen_features)
        except Exception as e:
            logging.warning(f"FVD computation failed: {e}")
            metrics['fvd'] = None
        
        logging.info(f"Processed {frame_idx} frames total")
        
        return metrics


def evaluate_image_quality(generated_path: str, reference_path: str,
                         output_path: Optional[str] = None,
                         device: str = "auto",
                         max_frames: Optional[int] = None) -> Dict[str, Union[float, Dict]]:
    """
    Main function to evaluate image or video quality
    
    Args:
        generated_path: Path to generated image/video
        reference_path: Path to reference image/video
        output_path: Path to save metrics JSON (optional)
        device: Device for LPIPS computation
        
    Returns:
        Dictionary containing quality metrics
    """
    evaluator = QualityEvaluator(device=device)
    
    # Determine if we're dealing with images or videos
    is_video_gen = generated_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    is_video_ref = reference_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video_gen != is_video_ref:
        raise ValueError("Both files must be either images or videos")
    
    # Compute metrics
    if is_video_gen:
        logging.info("Evaluating video quality...")
        metrics = evaluator.evaluate_videos(generated_path, reference_path, max_frames=max_frames)
    else:
        logging.info("Evaluating image quality...")
        metrics = evaluator.evaluate_images(generated_path, reference_path)
    
    # Add metadata
    result = {
        'generated_path': generated_path,
        'reference_path': reference_path,
        'file_type': 'video' if is_video_gen else 'image',
        'metrics': metrics,
        'timestamp': str(datetime.now())
    }
    
    # Save to file if specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"Metrics saved to: {output_path}")
    
    return result


def infer_origin_path_from_output_dir(generated_path: str, output_dir: Optional[str], tag: str) -> Optional[str]:
    """Infer ori reference path from cubic output directory layout."""

    def _prompt_dir_candidates(name: str) -> list[str]:
        candidates = [name]
        # Strip known speed/param suffixes to find the base prompt_id used by ori.
        #   cubic:   _2x / _2.5x
        #            _a2 / _b4 / _a2_b4
        #   tocache: _fr0.1
        #   pab:     _skip5
        #   jano:    _si2_mi4
        _suffix_patterns = [
            r'_[0-9]+(?:\.[0-9]+)?x$',
            r'_a[0-9]+(?:\.[0-9]+)?_b[0-9]+(?:\.[0-9]+)?$',
            r'_a[0-9]+(?:\.[0-9]+)?$',
            r'_b[0-9]+(?:\.[0-9]+)?$',
            r'_fr[0-9]+(?:\.[0-9]+)?$',
            r'_skip[0-9]+$',
            r'_si[0-9]+_mi[0-9]+$',
        ]

        current = name
        while True:
            stripped_any = False
            for pat in _suffix_patterns:
                stripped = re.sub(pat, '', current)
                if stripped and stripped != current:
                    if stripped not in candidates:
                        candidates.append(stripped)
                    current = stripped
                    stripped_any = True
                    break
            if not stripped_any:
                break
        return candidates

    gen_abs = os.path.abspath(generated_path)
    filename = os.path.basename(gen_abs)
    ori_filename = filename.replace(f"{tag}_", "ori_", 1)
    for _prefix in ("cubic_", "teacache_", "tocache_", "pab_", "jano_"):
        if ori_filename == filename and filename.startswith(_prefix):
            ori_filename = filename.replace(_prefix, "ori_", 1)
            break

    candidates = []
    if output_dir:
        out_abs = os.path.abspath(output_dir)
        prompt_dir = os.path.basename(out_abs)
        prompt_dir_candidates = _prompt_dir_candidates(prompt_dir)
        method_dir = os.path.basename(os.path.dirname(out_abs))
        model_root = os.path.dirname(os.path.dirname(out_abs))

        if method_dir in ("cubic", "teacache", "tocache", "pab", "jano"):
            for prompt_candidate in prompt_dir_candidates:
                candidates.append(os.path.join(model_root, "ori", prompt_candidate, ori_filename))

        for _method_token in ("cubic", "teacache", "tocache", "pab", "jano"):
            token = f"{os.sep}{_method_token}{os.sep}"
            if token in out_abs:
                for prompt_candidate in prompt_dir_candidates:
                    ori_dir = os.path.join(model_root, "ori", prompt_candidate)
                    candidates.append(os.path.join(ori_dir, ori_filename))

    for _method_token in ("cubic", "teacache", "tocache", "pab", "jano"):
        token = f"{os.sep}{_method_token}{os.sep}"
        if token in gen_abs:
            ori_abs = gen_abs.replace(token, f"{os.sep}ori{os.sep}")
            candidates.append(os.path.join(os.path.dirname(ori_abs), ori_filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate_quality_with_origin(
    image_path: str,
    tag: str,
    save_metrics: bool = True,
    baseline_path: str = None,
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
):
    """评测生成图像质量
    
    Args:
        image_path: 生成文件路径
        tag: 当前方法的标签（用于构造 baseline 路径和输出文件名）
        save_metrics: 是否将指标单独保存为 JSON 文件。设为 False 时只返回结果，
                      由调用方统一写入 params_metrics.json。
        baseline_path: 显式指定 baseline 文件路径。若为 None，默认按 tag 前缀替换。
    """
    
    # 构建基准文件路径
    if baseline_path is None:
        baseline_path = infer_origin_path_from_output_dir(
            generated_path=image_path,
            output_dir=output_dir,
            tag=tag,
        )
        if baseline_path is None:
            baseline_path = image_path.replace(f"{tag}_", "ori_")  # 默认同目录前缀替换
    
    if not os.path.exists(baseline_path):
        logging.error(f"Baseline file not found: {baseline_path}")
        logging.error("Please ensure baseline image exists for quality evaluation.")
        return None
    
    # 构建质量评测结果保存路径（仅在 save_metrics=True 时实际写入）
    metrics_path = os.path.join(os.path.dirname(image_path), f"{tag}_quality_metrics.json") if save_metrics else None
    
    try:
        # 执行质量评测
        logging.info(f"Evaluating quality against baseline: {baseline_path}")
        result = evaluate_image_quality(
            image_path,
            baseline_path,
            metrics_path,
            max_frames=max_frames,
        )

        
        # 打印结果
        metrics = result['metrics']
        logging.info(f"Quality Metrics:")
        if metrics.get('psnr') is not None:
            logging.info(f"  PSNR: {metrics['psnr']:.4f} dB")
        if metrics.get('ssim') is not None:
            logging.info(f"  SSIM: {metrics['ssim']:.4f}")
        if metrics.get('lpips') is not None:
            logging.info(f"  LPIPS: {metrics['lpips']:.4f}")
        if metrics.get('fid') is not None:
            logging.info(f"  FID: {metrics['fid']:.4f}")
        if metrics.get('fvd') is not None:
            logging.info(f"  FVD: {metrics['fvd']:.4f}")
        
        return result
        
    except Exception as e:
        logging.error(f"Quality evaluation failed: {e}")
        return None


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Image/Video Quality Evaluation Tool')
    
    parser.add_argument('--generated', '-g', type=str, required=True,
                        help='Path to generated image/video')
    parser.add_argument('--reference', '-r', type=str, required=True,
                        help='Path to reference image/video')
    parser.add_argument('--output', '-o', type=str, 
                        help='Path to save metrics JSON file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device for LPIPS computation')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process for video evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Evaluate quality
        result = evaluate_image_quality(
            args.generated, 
            args.reference, 
            args.output, 
            args.device,
            args.max_frames,
        )
        
        # Print results
        print("\n" + "="*60)
        print("QUALITY EVALUATION RESULTS")
        print("="*60)
        print(f"Generated: {args.generated}")
        print(f"Reference: {args.reference}")
        print(f"File Type: {result['file_type']}")
        print()
        
        metrics = result['metrics']
        print("Metrics:")
        if metrics.get('psnr') is not None:
            print(f"  PSNR: {metrics['psnr']:.4f} dB")
        if metrics.get('ssim') is not None:
            print(f"  SSIM: {metrics['ssim']:.4f}")
        if metrics.get('lpips') is not None:
            print(f"  LPIPS: {metrics['lpips']:.4f}")
        else:
            print("  LPIPS: Not available (install with: pip install lpips)")
        if metrics.get('fid') is not None:
            print(f"  FID: {metrics['fid']:.4f}")
        else:
            print("  FID: Not available")
        if metrics.get('fvd') is not None:
            print(f"  FVD: {metrics['fvd']:.4f}")
        elif result['file_type'] == 'video':
            print("  FVD: Not available")
        
        print("="*60)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
