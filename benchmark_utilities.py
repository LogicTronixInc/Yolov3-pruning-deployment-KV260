# benchmark_cpu_suite.py
import os, time, tempfile, copy, gc
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

# Optional ONNX Runtime (rows 5 & 6)
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader,
        QuantType, CalibrationMethod, QuantFormat,
    )
    HAS_ORT = True
except Exception:
    HAS_ORT = False


# ----------------------------
# Memory-format helpers
# ----------------------------
def to_channels_last_safe(model: nn.Module) -> nn.Module:
    """Mark module NHWC; only reorder 4D tensors to channels_last (safe)."""
    model.to(memory_format=torch.channels_last)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 4:
                p.data = p.data.contiguous(memory_format=torch.channels_last)
            else:
                p.data = p.data.contiguous()
        for b in model.buffers():
            if b.dim() == 4:
                b.data = b.contiguous(memory_format=torch.channels_last)
            else:
                b.data = b.contiguous()
    return model


# ----------------------------
# Benchmarking primitives
# ----------------------------
@torch.no_grad()
def bench_model(model: nn.Module, input_size: int, runs: int = 50, warmup: int = 10,
                channels_last: bool = False, device: str = "cpu") -> float:
    model.eval().to(device)
    x = torch.randn(1, 3, input_size, input_size, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    dt = time.perf_counter() - t0
    return runs / dt


@torch.no_grad()
def script_model(model: nn.Module, input_size: int, channels_last: bool = False,
                 device: str = "cpu") -> torch.jit.ScriptModule:
    model.eval().to(device)
    ex = torch.randn(1, 3, input_size, input_size, device=device)
    if channels_last:
        to_channels_last_safe(model)
        ex = ex.to(memory_format=torch.channels_last)
    scripted = torch.jit.trace(model, ex)
    scripted = torch.jit.optimize_for_inference(scripted)
    return scripted


# ----------------------------
# ONNX export / quant / bench
# ----------------------------
def export_onnx_fp32(model: nn.Module, onnx_path: str, input_size: int = 416,
                     input_name: str = "images", output_name: str = "preds") -> str:
    model.eval()
    ex = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model, ex, onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=None  # fixed shapes for apples-to-apples timing
    )
    return onnx_path


class ValCalibrationReader(CalibrationDataReader):
    """
    Pulls images from your val_loader for static INT8 calibration.
    Assumes the *second* element in each batch is the image tensor
    (as you previously iterated: for _, imgs, _, _ in loader).
    Falls back to the first 4D tensor if needed.
    """
    def __init__(self, val_loader, input_name="images", max_batches: int = 100):
        self.val_loader = val_loader
        self.input_name = input_name
        self.max_batches = max_batches
        self._reset()

    def _reset(self):
        self._it = iter(self.val_loader)
        self._n = 0

    def rewind(self):
        self._reset()

    def _extract_imgs(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
                imgs = batch[1]
            elif isinstance(batch[0], torch.Tensor):
                imgs = batch[0]
            else:
                imgs = None
                for x in batch:
                    if isinstance(x, torch.Tensor) and x.dim() == 4:
                        imgs = x; break
                if imgs is None:
                    raise RuntimeError("ValCalibrationReader: no 4D tensor found in batch.")
        elif isinstance(batch, torch.Tensor):
            imgs = batch
        else:
            raise RuntimeError("ValCalibrationReader: unsupported batch type.")
        if imgs.dim() != 4 or imgs.size(1) != 3:
            raise RuntimeError(f"Expected images [N,3,H,W], got {tuple(imgs.shape)}")
        return imgs

    def get_next(self):
        if self._n >= self.max_batches:
            return None
        try:
            batch = next(self._it)
        except StopIteration:
            return None
        imgs = self._extract_imgs(batch).detach().cpu().numpy().astype("float32")
        self._n += 1
        return {self.input_name: imgs}


def quantize_onnx_static_int8(fp32_path: str, int8_path: str, reader: CalibrationDataReader) -> str:
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        activation_type=QuantType.QUInt8,        # activations uint8
        weight_type=QuantType.QInt8,             # weights int8, per-channel if enabled
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
        quant_format=QuantFormat.QDQ,            # Q/DQ nodes around Conv/MatMul
        op_types_to_quantize=['Conv', 'MatMul']
    )
    return int8_path


def bench_ort_session(sess: 'ort.InferenceSession', input_size: int = 416,
                      runs: int = 50, warmup: int = 10, input_name: str = "images") -> float:
    import numpy as np
    x = np.random.randn(1, 3, input_size, input_size).astype("float32")
    feed = {input_name: x}
    for _ in range(warmup):
        _ = sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    return runs / (time.perf_counter() - t0)


# ----------------------------
# Pretty table writer
# ----------------------------
def write_table(path: str, rows: List[Dict[str, str]]):
    headers = ["Variant", "FPS", "Latency(ms)", "Speedup(x)", "%FPS"]
    w = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    line = "+".join("-" * (w[h] + 2) for h in headers)
    with open(path, "w") as f:
        def out(s): f.write(s + "\n"); print(s)
        out(line)
        out("| " + " | ".join(h.ljust(w[h]) for h in headers) + " |")
        out(line)
        for r in rows:
            vals = [str(r.get(h, "")) for h in headers]
            out("| " + " | ".join(vals[i].rjust(w[headers[i]]) for i in range(len(headers))) + " |")
        out(line)


# ----------------------------
# Suite
# ----------------------------
def run_cpu_bench_suite_full(model_fused: nn.Module,
                             val_loader,                     # for INT8 calibration
                             input_size: int = 416,
                             runs: int = 50,
                             warmup: int = 10,
                             log_txt: str = "cpu_bench_results.txt",
                             set_grid_xy: bool = True) -> List[Dict[str, str]]:
    """
    Benchmarks & logs:
      1) Eager (PyTorch)
      2) TorchScript
      3) TorchScript + channels_last
      4) TorchScript + channels_last + tuned *intra-op* threads (inter-op must be set once in main)
      5) ONNX Runtime (fp32)
      6) ONNX Runtime (static INT8, calibrated)
    """
    rows: List[Dict[str, str]] = []

    def _prep(m: nn.Module) -> nn.Module:
        m.eval()
        if set_grid_xy and hasattr(m, "set_grid_xy"):
            m.set_grid_xy(input_size)
        return m

    # 1) Eager
    m = _prep(copy.deepcopy(model_fused))
    fps_base = bench_model(m, input_size, runs, warmup, channels_last=False)
    rows.append({
        "Variant": "1) Normal (eager)",
        "FPS": f"{fps_base:.2f}",
        "Latency(ms)": f"{1000.0 / fps_base:.2f}",
        "Speedup(x)": f"{1.000:.3f}",
        "%FPS": f"{0.00:+.2f}%"
    })
    del m; gc.collect()

    # 2) TorchScript
    m = _prep(copy.deepcopy(model_fused))
    sm = script_model(m, input_size, channels_last=False)
    fps_script = bench_model(sm, input_size, runs, warmup, channels_last=False)
    rows.append({
        "Variant": "2) TorchScript",
        "FPS": f"{fps_script:.2f}",
        "Latency(ms)": f"{1000.0 / fps_script:.2f}",
        "Speedup(x)": f"{(fps_script / fps_base):.3f}",
        "%FPS": f"{(fps_script / fps_base - 1) * 100:+.2f}%"
    })
    del sm, m; gc.collect()

    # 3) TorchScript + channels_last
    m = _prep(copy.deepcopy(model_fused))
    sm = script_model(m, input_size, channels_last=True)
    fps_script_cl = bench_model(sm, input_size, runs, warmup, channels_last=True)
    rows.append({
        "Variant": "3) Script + channels_last",
        "FPS": f"{fps_script_cl:.2f}",
        "Latency(ms)": f"{1000.0 / fps_script_cl:.2f}",
        "Speedup(x)": f"{(fps_script_cl / fps_base):.3f}",
        "%FPS": f"{(fps_script_cl / fps_base - 1) * 100:+.2f}%"
    })
    del sm, m; gc.collect()

    # 4) TorchScript + channels_last + tuned intra-op threads
    m = _prep(copy.deepcopy(model_fused))
    sm = script_model(m, input_size, channels_last=True)
    prev_intra = torch.get_num_threads()
    try:
        # Use physical cores if available; don't change inter-op here (set it once in main)
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except Exception:
            phys = os.cpu_count() or 1
        torch.set_num_threads(int(phys))
        fps_thr = bench_model(sm, input_size, runs, warmup, channels_last=True)
    finally:
        torch.set_num_threads(prev_intra)
    rows.append({
        "Variant": "4) Script + CL + threads",
        "FPS": f"{fps_thr:.2f}",
        "Latency(ms)": f"{1000.0 / fps_thr:.2f}",
        "Speedup(x)": f"{(fps_thr / fps_base):.3f}",
        "%FPS": f"{(fps_thr / fps_base - 1) * 100:+.2f}%"
    })
    del sm, m; gc.collect()

    # 5) ONNX Runtime (fp32)
    if HAS_ORT:
        with tempfile.TemporaryDirectory() as td:
            fp32_p = os.path.join(td, "model_fp32.onnx")
            m = _prep(copy.deepcopy(model_fused))
            export_onnx_fp32(m, fp32_p, input_size=input_size, input_name="images", output_name="preds")
            sess_fp32 = ort.InferenceSession(fp32_p, providers=["CPUExecutionProvider"])
            fps_ort_fp32 = bench_ort_session(sess_fp32, input_size=input_size, runs=runs, warmup=warmup, input_name="images")
            try:
                sess_fp32._sess.close()
            except Exception:
                pass
        rows.append({
            "Variant": "5) ONNX Runtime (fp32)",
            "FPS": f"{fps_ort_fp32:.2f}",
            "Latency(ms)": f"{1000.0 / fps_ort_fp32:.2f}",
            "Speedup(x)": f"{(fps_ort_fp32 / fps_base):.3f}",
            "%FPS": f"{(fps_ort_fp32 / fps_base - 1) * 100:+.2f}%"
        })
        del m; gc.collect()

        # 6) ONNX Runtime (static INT8, calibrated)
        try:
            with tempfile.TemporaryDirectory() as td:
                fp32_p = os.path.join(td, "model_fp32.onnx")
                int8_p = os.path.join(td, "model_int8.onnx")
                m = _prep(copy.deepcopy(model_fused))
                export_onnx_fp32(m, fp32_p, input_size=input_size, input_name="images", output_name="preds")
                reader = ValCalibrationReader(val_loader, input_name="images", max_batches=100)
                quantize_onnx_static_int8(fp32_p, int8_p, reader)
                sess_int8 = ort.InferenceSession(int8_p, providers=["CPUExecutionProvider"])
                fps_ort_int8 = bench_ort_session(sess_int8, input_size=input_size, runs=runs, warmup=warmup, input_name="images")
                try:
                    sess_int8._sess.close()
                except Exception:
                    pass
            rows.append({
                "Variant": "6) ONNX Runtime (static INT8)",
                "FPS": f"{fps_ort_int8:.2f}",
                "Latency(ms)": f"{1000.0 / fps_ort_int8:.2f}",
                "Speedup(x)": f"{(fps_ort_int8 / fps_base):.3f}",
                "%FPS": f"{(fps_ort_int8 / fps_base - 1) * 100:+.2f}%"
            })
        except Exception as e:
            rows.append({
                "Variant": "6) ONNX Runtime (static INT8)",
                "FPS": "N/A", "Latency(ms)": "", "Speedup(x)": "",
                "%FPS": f"(INT8 failed: {type(e).__name__})"
            })
        del m; gc.collect()
    else:
        rows.append({
            "Variant": "5) ONNX Runtime (fp32)",
            "FPS": "N/A", "Latency(ms)": "", "Speedup(x)": "", "%FPS": "(onnxruntime not installed)"
        })
        rows.append({
            "Variant": "6) ONNX Runtime (static INT8)",
            "FPS": "N/A", "Latency(ms)": "", "Speedup(x)": "", "%FPS": "(onnxruntime not installed)"
        })

    os.makedirs(os.path.dirname(log_txt) or ".", exist_ok=True)
    write_table(log_txt, rows)
    return rows


# # benchmark_cpu_suite_stable.py
# import os, sys, time, json, tempfile, subprocess, shutil, statistics
# from typing import Dict, List, Tuple

# # ------------- utilities -------------
# def _cpu_affinity_list():
#     # Use physical cores if possible; fallback to all logical cores
#     try:
#         import psutil
#         phys = psutil.cpu_count(logical=False) or os.cpu_count() or 1
#         # pick first 'phys' logical CPU ids
#         return list(range(phys))
#     except Exception:
#         return list(range(os.cpu_count() or 1))

# def _run_python(code: str, env: Dict[str,str], workdir: str) -> Tuple[int,str,str]:
#     proc = subprocess.Popen(
#         [sys.executable, "-c", code],
#         cwd=workdir,
#         env=env,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#     )
#     out, err = proc.communicate()
#     return proc.returncode, out, err

# def _median_of_trials(measures: List[float]) -> float:
#     if not measures: return float("nan")
#     return statistics.median(measures)

# # ------------- child templates -------------
# # Each child process prints a single JSON line: {"fps": float}

# _CHILD_TEMPLATE_TORCH = r"""
# import os, time, torch, json, random
# torch.set_num_interop_threads({INTER})   # set before ops
# torch.set_num_threads({INTRA})
# random.seed(0); torch.manual_seed(0)
# from importlib import import_module

# # user model builder import path / symbol
# mod = import_module("{MODEL_MODULE}")
# build = getattr(mod, "{BUILD_FN}")

# # user model args
# input_size = {INPUT_SIZE}
# num_classes = {NUM_CLASSES}
# anchors = {ANCHORS}
# model_type = "{MODEL_TYPE}"
# ckpt = r"{CKPT}"

# model = build(input_size, num_classes, anchors, model_type, ckpt)
# if hasattr(model, "set_grid_xy"):
#     model.set_grid_xy(input_size)

# # optionally, apply user-provided fusion function
# try:
#     fuse = getattr(mod, "fuse_model")
#     fuse(model)
# except Exception:
#     pass

# # wrap channels_last?
# CL = {CHANNELS_LAST}
# if CL:
#     model.to(memory_format=torch.channels_last)

# # bench
# @torch.no_grad()
# def bench(m, runs={RUNS}, warmup={WARMUP}):
#     m.eval()
#     x = torch.randn(1,3,input_size,input_size)
#     if CL:
#         x = x.to(memory_format=torch.channels_last)
#     for _ in range(warmup): _ = m(x)
#     t0 = time.perf_counter()
#     for _ in range(runs): _ = m(x)
#     dt = time.perf_counter() - t0
#     return runs/dt

# # TorchScript?
# TS = {TORCHSCRIPT}
# if TS:
#     ex = torch.randn(1,3,input_size,input_size)
#     if CL: ex = ex.to(memory_format=torch.channels_last)
#     with torch.inference_mode():
#         model = torch.jit.trace(model, ex)
#         model = torch.jit.optimize_for_inference(model)

# fps = bench(model)
# print(json.dumps({"fps": fps}))
# """

# _CHILD_TEMPLATE_ONNX = r"""
# import os, time, json, random, tempfile
# import numpy as np
# import onnxruntime as ort
# import torch
# from importlib import import_module

# # threads before session creation
# so = ort.SessionOptions()
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# so.intra_op_num_threads = {INTRA}
# so.inter_op_num_threads = {INTER}
# so.enable_mem_pattern = True
# so.enable_cpu_mem_arena = True

# # build model
# mod = import_module("{MODEL_MODULE}")
# build = getattr(mod, "{BUILD_FN}")
# input_size = {INPUT_SIZE}
# num_classes = {NUM_CLASSES}
# anchors = {ANCHORS}
# model_type = "{MODEL_TYPE}"
# ckpt = r"{CKPT}"
# model = build(input_size, num_classes, anchors, model_type, ckpt)
# if hasattr(model, "set_grid_xy"):
#     model.set_grid_xy(input_size)
# try:
#     fuse = getattr(mod, "fuse_model"); fuse(model)
# except Exception:
#     pass
# model.eval()

# # export fp32
# with tempfile.TemporaryDirectory() as td:
#     fp32 = os.path.join(td, "m.onnx")
#     ex = torch.randn(1,3,input_size,input_size)
#     torch.onnx.export(model, ex, fp32, opset_version=13,
#                       do_constant_folding=True,
#                       input_names=["images"], output_names=["preds"],
#                       dynamic_axes=None)
#     providers = {PROVIDERS}
#     sess = ort.InferenceSession(fp32, sess_options=so, providers=providers)
#     x = np.random.randn(1,3,input_size,input_size).astype("float32")
#     feed = {"images": x}
#     for _ in range({WARMUP}): _ = sess.run(None, feed)
#     t0 = time.perf_counter()
#     for _ in range({RUNS}): _ = sess.run(None, feed)
#     fps = {RUNS} / (time.perf_counter() - t0)
# print(json.dumps({"fps": fps}))
# """

# # ------------- public API -------------
# def run_cpu_bench_suite_stable(
#     # you provide these to match your codebase
#     model_module: str,            # e.g. "experiment_entry" where build_model lives
#     build_fn: str,                # e.g. "build_model"
#     build_kwargs: Dict,           # dict with keys: input_size, num_classes, anchors, model_type, ckpt
#     trials: int = 5,              # total trials per variant
#     discard_first: int = 1,       # drop-cold trials
#     warmup: int = 30,
#     runs: int = 100,
#     intra_choices: List[int] = None,  # for the “threads” variant
#     onnx_providers: List[str] = None, # e.g. ["DnnlExecutionProvider","CPUExecutionProvider"]
#     log_txt: str = "cpu_bench_stable.txt",
# ) -> List[Dict[str,str]]:
#     """
#     Executes each variant in a fresh subprocess with fixed env/threads/affinity,
#     repeats trials, takes median (after discarding first cold trial).
#     """
#     onnx_providers = onnx_providers or ["CPUExecutionProvider"]
#     intra_choices = intra_choices or [1,2,4,8]
#     rows = []

#     # Base env: pin inter-op to 1, cap OpenMP/MKL, pin CPU affinity
#     base_env = os.environ.copy()
#     base_env["OMP_NUM_THREADS"] = base_env.get("OMP_NUM_THREADS", "1")
#     base_env["MKL_NUM_THREADS"] = base_env.get("MKL_NUM_THREADS", "1")
#     base_env["KMP_AFFINITY"] = base_env.get("KMP_AFFINITY", "granularity=fine,compact,1,0")
#     # Affinity: use taskset via subprocess to pin; or rely on KMP_AFFINITY.

#     def _variant(name: str, torchscript: bool, channels_last: bool, intra: int, inter: int) -> float:
#         code = _CHILD_TEMPLATE_TORCH.format(
#             MODEL_MODULE=model_module,
#             BUILD_FN=build_fn,
#             INPUT_SIZE=build_kwargs["input_size"],
#             NUM_CLASSES=build_kwargs["num_classes"],
#             ANCHORS=json.dumps(build_kwargs["anchors"]),
#             MODEL_TYPE=build_kwargs["model_type"],
#             CKPT=build_kwargs["ckpt"],
#             TORCHSCRIPT="True" if torchscript else "False",
#             CHANNELS_LAST="True" if channels_last else "False",
#             INTRA=intra,
#             INTER=inter,
#             RUNS=runs,
#             WARMUP=warmup,
#         )
#         measures = []
#         with tempfile.TemporaryDirectory() as td:
#             env = base_env.copy()
#             # These affect intra-op in PyTorch backends too
#             env["OMP_NUM_THREADS"] = str(intra)
#             env["MKL_NUM_THREADS"] = str(intra)
#             env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
#             # Repeat trials
#             for t in range(trials):
#                 rc, out, err = _run_python(code, env, td)
#                 if rc != 0:
#                     # surface error
#                     raise RuntimeError(f"[{name}] child failed: {err.strip()}")
#                 fps = json.loads(out.strip())["fps"]
#                 measures.append(fps)
#                 time.sleep(0.5)  # cool-down
#         # discard cold trials and return median
#         keep = measures[discard_first:] if len(measures) > discard_first else measures
#         return _median_of_trials(keep)

#     def _variant_onnx(name: str, intra: int, inter: int) -> float:
#         code = _CHILD_TEMPLATE_ONNX.format(
#             MODEL_MODULE=model_module,
#             BUILD_FN=build_fn,
#             INPUT_SIZE=build_kwargs["input_size"],
#             NUM_CLASSES=build_kwargs["num_classes"],
#             ANCHORS=json.dumps(build_kwargs["anchors"]),
#             MODEL_TYPE=build_kwargs["model_type"],
#             CKPT=build_kwargs["ckpt"],
#             INTRA=intra,
#             INTER=inter,
#             PROVIDERS=json.dumps(onnx_providers),
#             RUNS=runs,
#             WARMUP=warmup,
#         )
#         measures = []
#         with tempfile.TemporaryDirectory() as td:
#             env = base_env.copy()
#             env["OMP_NUM_THREADS"] = str(intra)
#             env["MKL_NUM_THREADS"] = str(intra)
#             env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
#             for t in range(trials):
#                 rc, out, err = _run_python(code, env, td)
#                 if rc != 0:
#                     raise RuntimeError(f"[{name}] child failed: {err.strip()}")
#                 fps = json.loads(out.strip())["fps"]
#                 measures.append(fps)
#                 time.sleep(0.5)
#         keep = measures[discard_first:] if len(measures) > discard_first else measures
#         return _median_of_trials(keep)

#     # 1) Normal (eager)
#     eager_fps = _variant("eager", torchscript=False, channels_last=False, intra=4, inter=1)
#     rows.append({"Variant":"1) Normal (eager)","FPS":f"{eager_fps:.2f}",
#                  "Latency(ms)":f"{1000.0/eager_fps:.2f}","Speedup(x)":"1.000","%FPS":"+0.00%"})

#     # 2) TorchScript
#     ts_fps = _variant("torchscript", torchscript=True, channels_last=False, intra=4, inter=1)
#     rows.append({"Variant":"2) TorchScript","FPS":f"{ts_fps:.2f}",
#                  "Latency(ms)":f"{1000.0/ts_fps:.2f}",
#                  "Speedup(x)":f"{ts_fps/eager_fps:.3f}",
#                  "%FPS":f"{(ts_fps/eager_fps-1)*100:+.2f}%"})

#     # 3) Script + channels_last
#     cl_fps = _variant("script+CL", torchscript=True, channels_last=True, intra=4, inter=1)
#     rows.append({"Variant":"3) Script + channels_last","FPS":f"{cl_fps:.2f}",
#                  "Latency(ms)":f"{1000.0/cl_fps:.2f}",
#                  "Speedup(x)":f"{cl_fps/eager_fps:.3f}",
#                  "%FPS":f"{(cl_fps/eager_fps-1)*100:+.2f}%"})

#     # 4) Script + CL + threads (sweep intra and pick best)
#     best_thr = 0.0
#     for n in intra_choices:
#         fps_n = _variant(f"script+CL+thr@{n}", torchscript=True, channels_last=True, intra=n, inter=1)
#         best_thr = max(best_thr, fps_n)
#     rows.append({"Variant":"4) Script + CL + threads","FPS":f"{best_thr:.2f}",
#                  "Latency(ms)":f"{1000.0/best_thr:.2f}",
#                  "Speedup(x)":f"{best_thr/eager_fps:.3f}",
#                  "%FPS":f"{(best_thr/eager_fps-1)*100:+.2f}%"})

#     # 5) ONNX Runtime (fp32)
#     try:
#         ort_fps = _variant_onnx("onnx-fp32", intra=4, inter=1)
#         rows.append({"Variant":"5) ONNX Runtime (fp32)","FPS":f"{ort_fps:.2f}",
#                      "Latency(ms)":f"{1000.0/ort_fps:.2f}",
#                      "Speedup(x)":f"{ort_fps/eager_fps:.3f}",
#                      "%FPS":f"{(ort_fps/eager_fps-1)*100:+.2f}%"})
#     except Exception as e:
#         rows.append({"Variant":"5) ONNX Runtime (fp32)","FPS":"N/A","Latency(ms)":"",
#                      "Speedup(x)":"","%FPS":f"({type(e).__name__})"})

#     # (Static INT8 omitted here to keep stable; add later once fp32 ORT is solid)

#     # write table
#     _write_table(log_txt, rows)
#     return rows


# def _write_table(path: str, rows: List[Dict[str,str]]):
#     headers = ["Variant","FPS","Latency(ms)","Speedup(x)","%FPS"]
#     w = {h:max(len(h),*(len(str(r.get(h,''))) for r in rows)) for h in headers}
#     line = "+".join("-"*(w[h]+2) for h in headers)
#     with open(path,"w") as f:
#         def out(s): f.write(s+"\n"); print(s)
#         out(line)
#         out("| " + " | ".join(h.ljust(w[h]) for h in headers) + " |")
#         out(line)
#         for r in rows:
#             vals = [str(r.get(h,"")) for h in headers]
#             out("| " + " | ".join(vals[i].rjust(w[headers[i]]) for i in range(len(headers))) + " |")
#         out(line)
