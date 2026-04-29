import argparse
from pathlib import Path
import torch
from statistics import mean

from flux2k_cubic import (
	TAG,
	load_pipeline,
	resolve_output_paths,
	run_selective_attention_comparison,
	run_straightness_sampling,
	save_straightness_summary,
)
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import save_params_and_metrics
from utils.timer import get_timer, get_time_statistics_dict, init_timer, print_time_statistics

PROMPT = "A cat holding a sign that says hello world"
NUM_IMAGES = 3
WARMUP_RUNS = 2

visualize_curvature = False  # 是否启用曲率可视化（会增加额外计算开销）
eval_quality = True
compare_attention_scenarios = False # 是否比较KV Cache引入的误差


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Flux2-klein cubic straightness visualization")
	parser.add_argument("--model_path", type=str, default=None, help="Model path or HuggingFace model ID")
	parser.add_argument("--prompt", type=str, default=PROMPT, help="Generation prompt")
	parser.add_argument(
		"--output_dir",
		type=Path,
		default=None,
		help="Output directory. If not set, uses results/flux2/{model_id}/cubic/{prompt_id}",
	)
	parser.add_argument(
		"--output_image",
		type=Path,
		default=None,
		help="Output image path. If not set, uses {output_dir}/{tag}_{prompt_id}_{model_id}.png",
	)
	parser.add_argument("--height", type=int, default=1024, help="Output image height")
	parser.add_argument("--width", type=int, default=1024, help="Output image width")
	parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
	parser.add_argument("--num_inference_steps", type=int, default=4, help="Inference steps")
	parser.add_argument("--seed", type=int, default=42, help="Base random seed")
	parser.add_argument("--num_images", type=int, default=NUM_IMAGES, help="Number of images/seeds to generate")
	parser.add_argument("--warmup_runs", type=int, default=WARMUP_RUNS, help="Warmup runs before timed generation")
	parser.add_argument("--device", type=str, default="cuda", help="Execution device")
	parser.add_argument("--max_sequence_length", type=int, default=512, help="Max text sequence length")
	parser.add_argument("--quality_eval", action="store_true", default=eval_quality, help="Enable quality evaluation against ori baseline")
	parser.add_argument(
		"--visualize_curvature",
		action="store_true",
		default=visualize_curvature,
		help="Enable cubic visualization export (partition/straightness)",
	)
	parser.add_argument(
		"--compare_attention_scenarios",
		action=argparse.BooleanOptionalAction,
		default=compare_attention_scenarios,
		help="Compare baseline/full attention outputs against selective no-kv and stale-kv scenarios",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	args.model_path = resolve_model_path(args.model_path, "--model_path")
	dtype = torch.bfloat16
	mode_suffix = "attention_compare" if bool(args.compare_attention_scenarios) else "selective"
	output_dir, output_image_path, model_id = resolve_output_paths(
		args.model_path, args.prompt, args.output_dir, args.output_image, mode_suffix=mode_suffix
	)
	output_dir.mkdir(parents=True, exist_ok=True)

	init_timer()
	with get_timer("load_pipeline"):
		pipe = load_pipeline(args.model_path, dtype, args.device)

	generator_device = pipe._execution_device.type
	seeds = [args.seed + i for i in range(max(1, int(args.num_images)))]
	print(
		"Inference config: "
		f"height={args.height}, width={args.width}, steps={args.num_inference_steps}, "
		f"guidance_scale={args.guidance_scale}, base_seed={args.seed}, "
		f"num_images={len(seeds)}, warmup_runs={args.warmup_runs}, "
		f"visualize_curvature={bool(args.visualize_curvature)}, "
		f"compare_attention_scenarios={bool(args.compare_attention_scenarios)}, "
		f"output_mode={mode_suffix}, generator_device={generator_device}",
		flush=True,
	)

	if args.compare_attention_scenarios:
		comparison_generator = torch.Generator(device=generator_device).manual_seed(args.seed)
		comparison_path = run_selective_attention_comparison(
			pipe=pipe,
			prompt=args.prompt,
			height=args.height,
			width=args.width,
			guidance_scale=args.guidance_scale,
			num_inference_steps=args.num_inference_steps,
			generator=comparison_generator,
			max_sequence_length=args.max_sequence_length,
			output_dir=output_dir,
		)
		print(f"Attention comparison written to {comparison_path}", flush=True)
		return

	# Warmup不计时：用于稳定首次运行开销，避免污染计时统计。
	for warmup_idx in range(max(0, int(args.warmup_runs))):
		warmup_generator = torch.Generator(device=generator_device).manual_seed(seeds[0] + warmup_idx)
		with torch.inference_mode():
			run_straightness_sampling(
				pipe=pipe,
				prompt=args.prompt,
				height=args.height,
				width=args.width,
				guidance_scale=args.guidance_scale,
				num_inference_steps=args.num_inference_steps,
				generator=warmup_generator,
				max_sequence_length=args.max_sequence_length,
				cubic_results_dir=None,
				enable_step_timing=False,
			)
	print(f"Warmup done: runs={max(0, int(args.warmup_runs))}", flush=True)

	image_paths: list[Path] = []
	per_seed_quality: dict[str, dict] = {}

	for seed in seeds:
		generator = torch.Generator(device=generator_device).manual_seed(seed)
		seed_output_dir = output_dir / f"seed{seed}"

		with get_timer("inference"):
			with torch.inference_mode():
				image, straightness = run_straightness_sampling(
					pipe=pipe,
					prompt=args.prompt,
					height=args.height,
					width=args.width,
					guidance_scale=args.guidance_scale,
					num_inference_steps=args.num_inference_steps,
					generator=generator,
					max_sequence_length=args.max_sequence_length,
					cubic_results_dir=seed_output_dir if args.visualize_curvature else None,
					enable_step_timing=True,
				)

		seed_image_path = output_image_path.with_name(f"{TAG}_{output_image_path.stem.replace(f'{TAG}_', '')}_seed{seed}.png")
		seed_image_path.parent.mkdir(parents=True, exist_ok=True)
		with get_timer("save_image"):
			image.save(seed_image_path)
		image_paths.append(seed_image_path)
		print(f"Image saved to: {seed_image_path.resolve()}", flush=True)

		if args.visualize_curvature:
			straightness.export_visualization(str(seed_output_dir))
			summary_path = save_straightness_summary(seed_output_dir, straightness)
			print(f"Straightness visualization exported to: {(seed_output_dir / 'straightness_vis').resolve()}", flush=True)
			print(f"Straightness summary saved to: {summary_path.resolve()}", flush=True)

		if args.quality_eval:
			quality_result = evaluate_quality_with_origin(
				image_path=str(seed_image_path),
				tag=TAG,
				save_metrics=False,
				baseline_path=None,
				output_dir=str(output_dir),
			)
			per_seed_quality[f"seed{seed}"] = quality_result["metrics"] if quality_result is not None else {}

	quality_metrics = None
	if args.quality_eval:
		valid_metrics = [m for m in per_seed_quality.values() if m]
		if valid_metrics:
			metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
			mean_metrics = {}
			for name in metric_names:
				values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
				mean_metrics[name] = mean(values) if values else None
			quality_metrics = {
				"per_seed": per_seed_quality,
				"mean": mean_metrics,
			}
		else:
			quality_metrics = {
				"per_seed": per_seed_quality,
				"mean": None,
			}

	print_time_statistics()

	params = {
		"model": "flux2",
		"method": TAG,
		"model_id": model_id,
		"model_path": args.model_path,
		"prompt": args.prompt,
		"output_images": [str(p) for p in image_paths],
		"base_seed": args.seed,
		"seeds": seeds,
		"num_images": len(seeds),
		"warmup_runs": max(0, int(args.warmup_runs)),
		"height": args.height,
		"width": args.width,
		"guidance_scale": args.guidance_scale,
		"num_inference_steps": args.num_inference_steps,
		"device": args.device,
		"straightness_records": len(straightness.records),
		"quality_eval": bool(args.quality_eval),
		"visualize_curvature": bool(args.visualize_curvature),
		"output_mode": mode_suffix,
	}
	params_path = save_params_and_metrics(
		str(output_dir),
		TAG,
		params,
		get_time_statistics_dict(),
		quality_metrics=quality_metrics,
	)
	print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
	main()
