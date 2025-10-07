import os
import sys
import json
import time
import gc
import argparse
import warnings
import random
import traceback
import contextlib
import numpy as np
import torch
import transformers
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

os.environ['NUMEXPR_MAX_THREADS'] = '64'
random.seed(42)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:98"

gpu_temp = os.environ.get("CUDA_VISIBLE_DEVICES")
gpu_id = gpu_temp[-1] if gpu_temp else "0"
print(f"Using GPU ID: {gpu_id}")

ENV_COMPRESSION_RATIO = float(os.environ.get("COMPRESSION_RATIO", 0.5))
ENV_PRESS_TYPE = os.environ.get("PRESS_TYPE", "knorm").lower()
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", 0))
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR", "Race_QwenKVPress_Results")
if SAMPLE_LIMIT > 0:
	print(f"Sample limit set to: {SAMPLE_LIMIT}")

print(f"KV Press config: compression_ratio={ENV_COMPRESSION_RATIO}, compression_type={ENV_PRESS_TYPE}")

sys.path.append("/data/to/your/code/path/Qwen_2.5")
try:
	from modeling_qwen2_5_omni_origin import Qwen2_5OmniForConditionalGeneration
	from processing_qwen2_5_omni import Qwen2_5OmniProcessor
	from qwen_omni_utils import process_mm_info
	QWEN_AVAILABLE = True
	print("[Info] Qwen2.5-Omni modules loaded successfully")
except ImportError as e:
	print(f"[Warning] Qwen2.5-Omni module import failed: {e}")
	QWEN_AVAILABLE = False
except Exception as e:
	print(f"[Warning] Qwen2.5-Omni module load error: {e}")
	QWEN_AVAILABLE = False

KV_PRESS_AVAILABLE = False
try:
	transformers_version = transformers.__version__
	print(f"[Info] Transformers version: {transformers_version}")
	try:
		from kvpress import (
			ExpectedAttentionPress,
			KnormPress,
			ObservedAttentionPress,
			RandomPress,
			SnapKVPress,
			StreamingLLMPress,
			TOVAPress,
		)
		KV_PRESS_AVAILABLE = True
		print("[Info] KV Press library loaded successfully")
	except ImportError as e:
		print(f"[Warning] KV Press library import failed: {e}")
		print("[Info] Attempting to import from local kvpress directory...")
		try:
			kvpress_path = os.path.join(os.path.dirname(__file__), '..', 'kvpress')
			if os.path.exists(kvpress_path):
				sys.path.insert(0, kvpress_path)
				from kvpress import (
					ExpectedAttentionPress,
					KnormPress,
					ObservedAttentionPress,
					RandomPress,
					SnapKVPress,
					StreamingLLMPress,
					TOVAPress,
				)
				KV_PRESS_AVAILABLE = True
				print("[Info] Loaded KV Press library from local directory successfully")
		except Exception as local_e:
			print(f"[Warning] Local KV Press library import also failed: {local_e}")
except Exception as e:
	print(f"[Warning] KV Press library load error: {e}")
	KV_PRESS_AVAILABLE = False

if not KV_PRESS_AVAILABLE:
	print("[Warning] KV Press library unavailable, running without compression (or exit in main based on args)")

_AUDIO_TOKEN_ID = 151646          # '<|AUDIO|>'
_AUDIO_BOS_TOKEN_ID = 151647      # '<|audio_bos|>'
_AUDIO_EOS_TOKEN_ID = 151648      # '<|audio_eos|>'

def parse_args():
	parser = argparse.ArgumentParser(description="RACE with Qwen2.5-Omni + KV Press")
	parser.add_argument("--model-path", type=str,
					   default="/data/to/your/model/path/Qwen2.5-Omni-3B")
	parser.add_argument("--data-root", type=str, default="/data/to/your/dataset/path/race_audio", help="RACE audio data root directory")
	parser.add_argument("--bench-file", type=str, default="/data/to/your/dataset/path/race_audio/race_benchmark.json", help="RACE benchmark JSON path")
	parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum generated token count")
	parser.add_argument("--min-seq-len", type=int, default=128, help="Compression threshold")
	parser.add_argument("--no-compress", action="store_true", help="Disable compression")
	return parser.parse_args()

def convert_to_serializable(obj):
	"""Recursively convert object to JSON serializable format (supports numpy/torch/custom objects)"""
	if isinstance(obj, dict):
		return {k: convert_to_serializable(v) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [convert_to_serializable(v) for v in obj]
	elif isinstance(obj, tuple):
		return tuple(convert_to_serializable(v) for v in obj)
	elif isinstance(obj, (np.integer, np.int64, np.int32)):
		return int(obj)
	elif isinstance(obj, (np.floating, np.float64, np.float32)):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	elif torch.is_tensor(obj):
		return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
	elif isinstance(obj, (np.bool_, bool)):
		return bool(obj)
	elif hasattr(obj, "__dict__"):
		return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
	else:
		return obj

def get_gpu_memory_usage():
	"""Get GPU memory usage"""
	if torch.cuda.is_available():
		allocated = torch.cuda.memory_allocated() / 1024**3  # GB
		reserved = torch.cuda.memory_reserved() / 1024**3    # GB
		return allocated, reserved
	return 0.0, 0.0

class RaceTimingStats:
	"""Track RACE inference timing statistics"""
	def __init__(self):
		self.timing_records = []
		self.total_samples = 0
		self.total_prefill_time = 0.0
		self.total_decode_time = 0.0
		self.total_tokens = 0

	def add_record(self, prefill_time, decode_time, output_tokens, input_tokens, audio_duration=None):
		self.total_samples += 1
		self.total_prefill_time += prefill_time
		self.total_decode_time += decode_time
		self.total_tokens += output_tokens

		record = {
			"prefill_time": prefill_time,
			"decode_time": decode_time,
			"total_time": prefill_time + decode_time,
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"audio_duration": audio_duration,
			"tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
		}
		self.timing_records.append(record)

	def get_summary(self):
		if self.total_samples == 0:
			return {}
		return {
			"total_samples": self.total_samples,
			"avg_prefill_time": self.total_prefill_time / self.total_samples,
			"avg_decode_time": self.total_decode_time / self.total_samples,
			"avg_total_time": (self.total_prefill_time + self.total_decode_time) / self.total_samples,
			"total_tokens": self.total_tokens,
			"avg_tokens": self.total_tokens / self.total_samples,
			"avg_tokens_per_sec": self.total_tokens / self.total_decode_time if self.total_decode_time > 0 else 0,
		}

	def export_to_json(self, output_file):
		result = {"summary": self.get_summary(), "detailed_records": self.timing_records}
		with open(output_file, "w", encoding="utf-8") as f:
			json.dump(convert_to_serializable(result), f, indent=2, ensure_ascii=False)
		return output_file

def clean_text_response(response: str) -> str:
	"""Extract the first A/B/C/D option label from model output"""
	if not response:
		return ""
	resp = response.strip().upper()
	for ch in resp:
		if ch in ["A", "B", "C", "D"]:
			return ch
	words = resp.split()
	for word in words:
		for ch in word:
			if ch in ["A", "B", "C", "D"]:
				return ch
	return words[0] if words else ""

def calculate_race_metrics(y_true, y_pred, subset_labels=None):
	"""Reuse DART's detailed metrics calculation"""
	valid_indices = []
	clean_y_true, clean_y_pred = [], []
	clean_subset_labels = [] if subset_labels is not None else None
	valid_labels = ["A", "B", "C", "D"]

	for i, (t, p) in enumerate(zip(y_true, y_pred)):
		if t in valid_labels and p in valid_labels:
			valid_indices.append(i)
			clean_y_true.append(t)
			clean_y_pred.append(p)
			if subset_labels is not None:
				clean_subset_labels.append(subset_labels[i])

	if len(clean_y_true) == 0:
		return {
			'accuracy': 0.0,
			'precision_macro': 0.0,
			'recall_macro': 0.0,
			'f1_macro': 0.0,
			'precision_weighted': 0.0,
			'recall_weighted': 0.0,
			'f1_weighted': 0.0,
			'per_class_metrics': {},
			'subset_metrics': {},
			'classification_report': "No valid predictions",
			'valid_samples': 0,
			'total_samples': len(y_true),
			'class_labels': valid_labels,
		}

	accuracy = accuracy_score(clean_y_true, clean_y_pred)
	precision, recall, f1, support = precision_recall_fscore_support(
		clean_y_true, clean_y_pred, labels=valid_labels, average=None, zero_division=0
	)
	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
		clean_y_true, clean_y_pred, labels=valid_labels, average='macro', zero_division=0
	)
	precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
		clean_y_true, clean_y_pred, labels=valid_labels, average='weighted', zero_division=0
	)

	per_class_metrics = {}
	for i, label in enumerate(valid_labels):
		per_class_metrics[label] = {
			'precision': float(precision[i]) if i < len(precision) else 0.0,
			'recall': float(recall[i]) if i < len(recall) else 0.0,
			'f1_score': float(f1[i]) if i < len(f1) else 0.0,
			'support': int(support[i]) if i < len(support) else 0,
		}

	subset_metrics = {}
	if clean_subset_labels is not None:
		unique_subsets = list(set(clean_subset_labels))
		for subset in unique_subsets:
			subset_indices = [i for i, label in enumerate(clean_subset_labels) if label == subset]
			if subset_indices:
				subset_true = [clean_y_true[i] for i in subset_indices]
				subset_pred = [clean_y_pred[i] for i in subset_indices]
				subset_accuracy = accuracy_score(subset_true, subset_pred)
				subset_precision, subset_recall, subset_f1, _ = precision_recall_fscore_support(
					subset_true, subset_pred, average='macro', zero_division=0
				)
				subset_metrics[subset] = {
					'accuracy': float(subset_accuracy),
					'precision': float(subset_precision),
					'recall': float(subset_recall),
					'f1_score': float(subset_f1),
					'samples': len(subset_indices),
				}

	report = classification_report(
		clean_y_true, clean_y_pred, labels=["A", "B", "C", "D"],
		target_names=[f"Choice {c}" for c in ["A", "B", "C", "D"]],
		zero_division=0, digits=4
	)

	return {
		'accuracy': float(accuracy),
		'precision_macro': float(precision_macro),
		'recall_macro': float(recall_macro),
		'f1_macro': float(f1_macro),
		'precision_weighted': float(precision_weighted),
		'recall_weighted': float(recall_weighted),
		'f1_weighted': float(f1_weighted),
		'per_class_metrics': per_class_metrics,
		'subset_metrics': subset_metrics,
		'classification_report': report,
		'valid_samples': len(clean_y_true),
		'total_samples': len(y_true),
		'class_labels': ["A", "B", "C", "D"],
	}

def patch_qwen_for_kvpress(model):
	"""Add rotary_emb compatibility for Qwen2.5-Omni"""
	try_paths = []
	if hasattr(model, 'thinker'):
		try_paths.append(model.thinker)
		if hasattr(model.thinker, 'model'):
			try_paths.append(model.thinker.model)
	try_paths.append(model)

	base = None
	for cand in try_paths:
		if hasattr(cand, 'model') and hasattr(cand.model, 'layers'):
			base = cand.model
			break
		if hasattr(cand, 'layers'):
			base = cand
			break
	if base is None:
		return False
	if hasattr(base, 'rotary_emb') and base.rotary_emb is not None:
		return True
	if hasattr(base, 'layers') and len(base.layers) > 0:
		first_layer = base.layers[0]
		for attr in ['attn', 'self_attn']:
			if hasattr(first_layer, attr):
				sub = getattr(first_layer, attr)
				if hasattr(sub, 'rotary_emb') and sub.rotary_emb is not None:
					base.rotary_emb = sub.rotary_emb
					return True
	try:
		hidden_size = model.config.hidden_size if hasattr(model, 'config') else 0
		num_heads = model.config.num_attention_heads if hasattr(model, 'config') else 1
		head_dim = hidden_size // num_heads if num_heads else 0
		class SimpleRotaryEmbedding:
			def __init__(self, dim, max_position_embeddings=32768, base=10000):
				self.dim = dim
				self.max_position_embeddings = max_position_embeddings
				self.base = base
		base.rotary_emb = SimpleRotaryEmbedding(dim=head_dim)
		return True
	except Exception:
		return False

def create_kvpress_adapter(model, press_obj):
	"""Create KV Press adapter for Qwen2.5-Omni (register attention forward hooks, compatible with self_attn/attn)"""
	class KVPressAdapter:
		def __init__(self, original_model, press_object):
			self.original_model = original_model
			self.press_object = press_object
			self.hooks = []
		def __enter__(self):
			hooks = []
			try:
				base_model = None
				if hasattr(self.original_model, 'thinker') and hasattr(self.original_model.thinker, 'model'):
					base_model = self.original_model.thinker.model
				elif hasattr(self.original_model, 'model'):
					base_model = self.original_model.model
				if base_model is None or not hasattr(base_model, 'layers'):
					self.hooks = []
					return self
				base_rotary = getattr(base_model, 'rotary_emb', None)
				ok = 0
				for layer in base_model.layers:
					attn = getattr(layer, 'self_attn', None)
					if attn is None:
						attn = getattr(layer, 'attn', None)
					if attn is None:
						continue
					if base_rotary is not None and (not hasattr(attn, 'rotary_emb') or getattr(attn, 'rotary_emb', None) is None):
						attn.rotary_emb = base_rotary
					if hasattr(self.press_object, 'forward_hook'):
						h = attn.register_forward_hook(self.press_object.forward_hook, with_kwargs=True)
						hooks.append(h)
						ok += 1
				self.hooks = hooks
				print(f"[KVPressAdapter] Registered hooks: {ok}/{len(getattr(base_model, 'layers', []))}")
				return self
			except Exception as e:
				print(f"[KVPressAdapter] Error: {e}")
				for hook in hooks:
					try:
						hook.remove()
					except Exception:
						pass
				self.hooks = []
				return self
		def __exit__(self, exc_type, exc_val, exc_tb):
			removed = 0
			for hook in getattr(self, 'hooks', []):
				try:
					hook.remove()
					removed += 1
				except Exception:
					pass
			if removed:
				print(f"[KVPressAdapter] Cleaned up hooks: {removed}")
	return KVPressAdapter(model, press_obj)

def verify_kvpress_compatibility(model, press_type):
	try:
		base_model = None
		if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
			base_model = model.thinker.model
		elif hasattr(model, 'model'):
			base_model = model.model
		if base_model is None or not hasattr(base_model, 'layers') or len(base_model.layers) == 0:
			return False
		if press_type in ['snap', 'tova']:
			global_rotary = getattr(base_model, 'rotary_emb', None)
			if global_rotary is None:
				return False
			for i in range(min(3, len(base_model.layers))):
				lyr = base_model.layers[i]
				attn = getattr(lyr, 'self_attn', None) or getattr(lyr, 'attn', None)
				if attn is None or getattr(attn, 'rotary_emb', None) is None:
					return False
		first = base_model.layers[0]
		attn0 = getattr(first, 'self_attn', None) or getattr(first, 'attn', None)
		if attn0 is None:
			return False
		for attr in ['q_proj', 'k_proj', 'v_proj']:
			if not hasattr(attn0, attr):
				return False
		return True
	except Exception:
		return False

def verify_tova_multimodal_compatibility(model):
	try:
		base_model = None
		if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
			base_model = model.thinker.model
		elif hasattr(model, 'model'):
			base_model = model.model
		if base_model is None:
			return False
		rotary = getattr(base_model, 'rotary_emb', None)
		if rotary is None:
			return False
		if hasattr(rotary, 'head_dim'):
			hd = rotary.head_dim
			if hd <= 0 or hd % 2 != 0:
				return False
		return True
	except Exception:
		return False

def verify_snapkv_multimodal_compatibility(model):
	try:
		base_model = None
		if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
			base_model = model.thinker.model
		elif hasattr(model, 'model'):
			base_model = model.model
		if base_model is None:
			return False
		rotary = getattr(base_model, 'rotary_emb', None)
		if rotary is None or not hasattr(rotary, 'forward'):
			return False
		inv = getattr(rotary, 'inv_freq', None)
		if inv is not None and hasattr(inv, 'shape'):
			if len(inv.shape) != 1 or inv.shape[0] == 0:
				return False
		return True
	except Exception:
		return False

def initialize_kv_press(model, press_type: str, compression_ratio: float, min_seq_len: int):
	"""Initialize KV Press compression (HAD simplified version)"""
	if not KV_PRESS_AVAILABLE:
		print("[Warning] KV Press unavailable, skipping compression initialization")
		return None

	print(f"Initializing KV Press: type={press_type}, compression_ratio={compression_ratio}")

	try:
		if press_type == 'expected':
			press_obj = ExpectedAttentionPress(compression_ratio=compression_ratio)
		elif press_type == 'observed':
			press_obj = ObservedAttentionPress(compression_ratio=compression_ratio)
		elif press_type == 'random':
			press_obj = RandomPress(compression_ratio=compression_ratio)
			print("[KVPress] Using RandomPress (compatible with all models)")
		elif press_type == 'streaming':
			press_obj = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=4)
		elif press_type == 'tova':
			press_obj = TOVAPress(compression_ratio=compression_ratio)
			print("[KVPress] Using TOVAPress")
		elif press_type == 'snap':
			press_obj = SnapKVPress(compression_ratio=compression_ratio)
			print("[KVPress] Using SnapKVPress")
		else:
			press_obj = KnormPress(compression_ratio=compression_ratio)
			print("[KVPress] Using default KnormPress (compatible with all models)")

		if hasattr(press_obj, 'min_seq_len'):
			try:
				setattr(press_obj, 'min_seq_len', min_seq_len)
			except Exception:
				pass

		print(f"[KVPress] Created {type(press_obj).__name__}")
		return press_obj
	except Exception as e:
		print(f"[KVPress] Failed to create {press_type} object: {e}")
		return None

def main():
	args = parse_args()

	result_dir = os.path.abspath(RESULTS_DIR_ENV)
	os.makedirs(result_dir, exist_ok=True)

	print(f"\n=== RACE KV Press Evaluation Config ===")
	print(f"GPU ID: {gpu_id}")
	print(f"KV Press compression type: {ENV_PRESS_TYPE}")
	print(f"Compression ratio: {ENV_COMPRESSION_RATIO}")
	print(f"Data directory: {args.data_root}")
	print(f"Benchmark file: {args.bench_file}")
	if SAMPLE_LIMIT > 0:
		print(f"Sample limit: {SAMPLE_LIMIT}")
	print("=" * 50)

	method_name = "no_compress" if args.no_compress else ENV_PRESS_TYPE
	ratio_str = f"{ENV_COMPRESSION_RATIO:.3f}"
	output_file = f"{result_dir}/race_results_kvpress_{method_name}_{ratio_str}.json"
	timing_output_file = f"{result_dir}/race_timing_stats_kvpress_{method_name}_{ratio_str}.json"
	print(f"Results will be saved to: {output_file}")
	print(f"Timing statistics will be saved to: {timing_output_file}")

	print("Loading Qwen2.5-Omni model...")
	model_path = args.model_path
	device_map = {"": 0}
	processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
	attn_impl = "flash_attention_2" if (args.no_compress) else "eager"
	model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
		model_path,
		device_map=device_map,
		torch_dtype=torch.bfloat16,
		attn_implementation=attn_impl,
		trust_remote_code=True,
	)
	try:
		if hasattr(model, 'disable_talker'):
			model.disable_talker()
	except Exception:
		pass
	print("Model loaded successfully")

	print("Configuring model audio parameters...")
	if hasattr(model, 'thinker') and hasattr(model.thinker, 'config'):
		cfg = model.thinker.config
		if not hasattr(cfg, 'audio_token_id'):
			cfg.audio_token_id = _AUDIO_TOKEN_ID
		if not hasattr(cfg, 'image_token_id'):
			cfg.image_token_id = 151655
		if not hasattr(cfg, 'video_token_id'):
			cfg.video_token_id = 151656
		if not hasattr(cfg, 'audio_bos_token_id'):
			cfg.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
		if not hasattr(cfg, 'audio_eos_token_id'):
			cfg.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
		if not hasattr(cfg, 'image_layer_idx'):
			cfg.image_layer_idx = False
		if not hasattr(cfg, 'audio_layer_idx'):
			cfg.audio_layer_idx = None
		if not hasattr(cfg, 'audio_token_num'):
			cfg.audio_token_num = None
		if not hasattr(cfg, 'audio_token_start'):
			cfg.audio_token_start = None
		if not hasattr(cfg, 'audio_prune_ratio'):
			cfg.audio_prune_ratio = 0
		if not hasattr(cfg, 'random'):
			cfg.random = False
		if not hasattr(cfg, 'frame'):
			cfg.frame = False
	if hasattr(model, 'thinker') and hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
		cfg2 = model.thinker.model.config
		if not hasattr(cfg2, 'audio_token_id'):
			cfg2.audio_token_id = _AUDIO_TOKEN_ID
		if not hasattr(cfg2, 'image_token_id'):
			cfg2.image_token_id = 151655
		if not hasattr(cfg2, 'video_token_id'):
			cfg2.video_token_id = 151656
		if not hasattr(cfg2, 'audio_bos_token_id'):
			cfg2.audio_bos_token_id = _AUDIO_BOS_TOKEN_ID
		if not hasattr(cfg2, 'audio_eos_token_id'):
			cfg2.audio_eos_token_id = _AUDIO_EOS_TOKEN_ID
		if not hasattr(cfg2, 'image_layer_idx'):
			cfg2.image_layer_idx = False
		if not hasattr(cfg2, 'audio_layer_idx'):
			cfg2.audio_layer_idx = None
		if not hasattr(cfg2, 'audio_token_num'):
			cfg2.audio_token_num = None
		if not hasattr(cfg2, 'audio_token_start'):
			cfg2.audio_token_start = None
		if not hasattr(cfg2, 'audio_prune_ratio'):
			cfg2.audio_prune_ratio = 0
		if not hasattr(cfg2, 'random'):
			cfg2.random = False
		if not hasattr(cfg2, 'frame'):
			cfg2.frame = False

	if args.no_compress:
		press_obj = None
		print("[KVPress] Compression disabled")
	else:
		press_obj = initialize_kv_press(model, ENV_PRESS_TYPE, ENV_COMPRESSION_RATIO, args.min_seq_len)
		if press_obj is None:
			print("[KVPress] Not enabled (creation failed or unavailable)")
		else:
			print(f"[KVPress] Enabled {ENV_PRESS_TYPE}, compression ratio: {ENV_COMPRESSION_RATIO}")

	bench_file = os.path.abspath(args.bench_file)
	if not os.path.exists(bench_file):
		print(f"Error: Benchmark file not found: {bench_file}")
		return
	with open(bench_file, 'r', encoding='utf-8') as f:
		benchmark = json.load(f)

	if SAMPLE_LIMIT > 0 and len(benchmark) > SAMPLE_LIMIT:
		benchmark = benchmark[:SAMPLE_LIMIT]
		print(f"Sample count limited to: {len(benchmark)}")

	initial_allocated, initial_reserved = get_gpu_memory_usage()
	print(f"After model load GPU memory - allocated: {initial_allocated:.2f}GB, reserved: {initial_reserved:.2f}GB")

	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()

	results = []
	timing_stats = RaceTimingStats()
	correct_count = 0
	peak_allocated_gb = 0.0
	peak_reserved_gb = 0.0

	tqdm_kwargs = {'ascii': True, 'dynamic_ncols': True, 'file': sys.stdout}
	print(f"Starting evaluation with {len(benchmark)} samples...")
	with tqdm(total=len(benchmark), desc="RACE Evaluation (Qwen2.5 KVPress)", position=0, leave=True, **tqdm_kwargs) as pbar:
		for idx, sample in enumerate(benchmark):
			try:
				audio_rel = sample.get("audio_path", "")
				audio_path = os.path.join(args.data_root, audio_rel)
				if not os.path.exists(audio_path):
					print(f"Warning: Audio file does not exist: {audio_path}")
					pbar.update(1)
					continue

				question = sample.get("question", "")
				options = sample.get("options", [])
				instruction = f"Question: {question}\n\nOptions:\n"
				for i_opt, opt in enumerate(options):
					letter = chr(65 + i_opt)
					instruction += f"{letter}. {opt}\n"
				instruction += "\nRespond with only the letter of the correct option (A, B, C, or D)."
				sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Listen to this audio of a passage being read aloud, then answer the multiple-choice question based solely on the information from the audio."

				messages = [
					{"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
					{"role": "user", "content": [
						{"type": "audio", "audio": audio_path},
						{"type": "text", "text": instruction},
					]},
				]

				audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
				text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
				if isinstance(text, list):
					text = text[0]
				inputs = processor(
					text=text, audio=audios, images=images, videos=videos,
					return_tensors="pt", padding=True, use_audio_in_video=True,
				)
				inputs = inputs.to(model.device)

				inputs = {k: (v.to(model.dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
						  for k, v in inputs.items()}

				audio_token_length = 0
				input_token_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
				if 'input_ids' in inputs:
					token_ids = inputs['input_ids'][0].tolist()
					if _AUDIO_BOS_TOKEN_ID in token_ids and _AUDIO_EOS_TOKEN_ID in token_ids:
						s = token_ids.index(_AUDIO_BOS_TOKEN_ID)
						e = token_ids.index(_AUDIO_EOS_TOKEN_ID)
						audio_token_length = e - s + 1

				prefill_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
				prefill_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
				if prefill_start:
					prefill_start.record()
				use_compression = bool(press_obj) and (input_token_length >= args.min_seq_len)
				with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
					_ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
				if prefill_end:
					prefill_end.record()

				gen_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
				gen_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
				if gen_start:
					gen_start.record()
				with torch.no_grad(), (create_kvpress_adapter(model, press_obj) if use_compression else contextlib.nullcontext()):
					out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
				if gen_end:
					gen_end.record()
				if torch.cuda.is_available():
					torch.cuda.synchronize()

				if torch.cuda.is_available():
					current_allocated = torch.cuda.max_memory_allocated() / (1024**3)
					current_reserved = torch.cuda.max_memory_reserved() / (1024**3)
					peak_allocated_gb = max(peak_allocated_gb, current_allocated)
					peak_reserved_gb = max(peak_reserved_gb, current_reserved)

				prefill_time = prefill_start.elapsed_time(prefill_end)/1000 if prefill_start else 0.0
				total_time = gen_start.elapsed_time(gen_end)/1000 if gen_start else 0.0
				decode_time = max(total_time - prefill_time, 0.0)

				output_text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
				if "assistant\n" in output_text:
					output_text = output_text.split("assistant\n")[-1].strip()
				predicted_choice = clean_text_response(output_text)
				ground_truth_choice = sample.get("answer", "")
				is_correct = predicted_choice == ground_truth_choice
				if is_correct:
					correct_count += 1

				output_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
				timing_stats.add_record(
					prefill_time=prefill_time,
					decode_time=decode_time,
					output_tokens=output_tokens,
					input_tokens=input_token_length,
					audio_duration=None,
				)

				difficulty = "high" if "high" in audio_rel else ("middle" if "middle" in audio_rel else "unknown")

				result = {
					"id": f"{sample.get('article_id', 'art')}_{sample.get('question_idx', idx)}",
					"question": question,
					"options": options,
					"correct_answer": ground_truth_choice,
					"predicted_answer": predicted_choice,
					"correct": is_correct,
					"difficulty": difficulty,
					"audio_path": audio_rel,
					"response_text": output_text,
					"gt": ground_truth_choice,
					"pred": predicted_choice,
					"subset": difficulty,
					"compression_info": {
						"enabled": bool(press_obj),
						"used": use_compression,
						"press_type": ENV_PRESS_TYPE if press_obj else None,
						"compression_ratio": ENV_COMPRESSION_RATIO if press_obj else None,
						"min_seq_len": args.min_seq_len,
						"input_tokens": input_token_length,
					},
					"timing": {
						"prefill_time": prefill_time,
						"decode_time": decode_time,
						"total_time": prefill_time + decode_time,
						"input_tokens": input_token_length,
						"output_tokens": output_tokens,
						"tokens_per_sec": output_tokens / decode_time if decode_time > 0 else 0,
						"audio_tokens": audio_token_length,
					},
				}
				results.append(result)

				current_acc = correct_count / (idx + 1)
				pbar.set_postfix({'Acc': f"{current_acc:.3f}", 'Tokens/s': f"{(output_tokens/decode_time):.1f}" if decode_time>0 else "N/A"})
				pbar.update(1)

				del inputs, out_ids
				torch.cuda.empty_cache()
			except Exception as e:
				print(f"\nError processing sample {idx}: {e}")
				traceback.print_exc()
				pbar.update(1)
				continue

	total = len(results)
	overall_acc = sum(r["correct"] for r in results) / total * 100 if total > 0 else 0
	y_true = [r["gt"] for r in results]
	y_pred = [r["pred"] for r in results]
	subset_labels = [r["subset"] for r in results]
	detailed_metrics = calculate_race_metrics(y_true, y_pred, subset_labels)

	summary = {
		"total_samples": total,
		"correct_samples": sum(r["correct"] for r in results),
		"overall_accuracy": overall_acc,
		"sklearn_metrics": detailed_metrics,
		"config": {
			"gpu_id": gpu_id,
			"model_path": args.model_path,
			"compression_enabled": not args.no_compress and (press_obj is not None),
			"press_type": ENV_PRESS_TYPE if press_obj else None,
			"compression_ratio": ENV_COMPRESSION_RATIO if press_obj else None,
			"sample_limit": SAMPLE_LIMIT,
		},
		"timing": timing_stats.get_summary(),
		"gpu_memory": {
			"initial_allocated_gb": initial_allocated,
			"initial_reserved_gb": initial_reserved,
			"peak_allocated_gb": peak_allocated_gb,
			"peak_reserved_gb": peak_reserved_gb,
			"device_id": gpu_id,
		},
	}

	final_results = {"summary": summary, "samples": results}

	print(f"Saving results to: {output_file}")
	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(convert_to_serializable(final_results), f, ensure_ascii=False, indent=2)

	print(f"Saving timing statistics to: {timing_output_file}")
	timing_stats.export_to_json(timing_output_file)

	print("\n=== RACE Evaluation Summary (Qwen2.5-Omni KVPress) ===")
	print(f"Total samples: {total}")
	print(f"Total Accuracy: {overall_acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
	metrics = detailed_metrics
	print("\n=== Detailed Evaluation Metrics (sklearn) ===")
	print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
	print(f"F1 score (macro average): {metrics.get('f1_macro', 0):.4f}")
	print(f"F1 score (weighted average): {metrics.get('f1_weighted', 0):.4f}")
	print(f"Precision (macro average): {metrics.get('precision_macro', 0):.4f}")
	print(f"Recall (macro average): {metrics.get('recall_macro', 0):.4f}")

	print("\n=== Subset Evaluation Metrics ===")
	for subset, subset_metrics in metrics.get('subset_metrics', {}).items():
		print(f"{subset.upper()} set: Accuracy={subset_metrics['accuracy']:.4f}, Precision={subset_metrics['precision']:.4f}, Recall={subset_metrics['recall']:.4f}, F1={subset_metrics['f1_score']:.4f}, Samples={subset_metrics['samples']}")

	print("\n=== Classification Detailed Report ===")
	print(metrics.get('classification_report', ''))

	print(f"\nResults saved to: {output_file}")
	print(f"Timing statistics saved to: {timing_output_file}")

if __name__ == "__main__":
	main()