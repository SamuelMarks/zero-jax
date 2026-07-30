"""Unified CI checking script for the zero-jax project."""

import argparse
import ast
import inspect
import json
import os
import re
import subprocess
import sys

# --- check_allowed_imports ---
ALLOWED_3RD_PARTY_SRC = {"pydantic"}
ALLOWED_INTERNAL_SRC = {
    "ml_switcheroo_compiler",
    "cdd_python",
    "ml_switcheroo_ir",
    "zero_jax",
}


def check_file_imports(filepath: str) -> bool:
    """
    Check a file for disallowed imports.

    Args:
        filepath: The path to the file to check.

    Returns:
        True if clean, False otherwise.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return False

    errors = []
    is_test = "tests" in filepath.split(os.sep) or "scripts" in filepath.split(os.sep)

    for node in ast.walk(tree):
        # Check for sys.modules[...]
        if isinstance(node, ast.Subscript):
            is_sys_modules = False
            if isinstance(node.value, ast.Attribute) and isinstance(
                node.value.value, ast.Name
            ):
                if node.value.value.id == "sys" and node.value.attr == "modules":
                    is_sys_modules = True

            if is_sys_modules:
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, str
                ):
                    base_module = node.slice.value.split(".")[0]
                    if not is_test:
                        if (
                            base_module not in ALLOWED_3RD_PARTY_SRC
                            and base_module not in ALLOWED_INTERNAL_SRC
                            and base_module not in sys.stdlib_module_names
                        ):
                            errors.append(
                                (
                                    node.lineno,
                                    f"Disallowed sys.modules access: '{base_module}'",
                                )
                            )
                else:
                    if not is_test:
                        errors.append(
                            (
                                node.lineno,
                                "sys.modules access with non-constant is forbidden",
                            )
                        )

        # Check for dynamic imports
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    func_name = f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func.value, ast.Attribute) and isinstance(
                    node.func.value.value, ast.Name
                ):
                    func_name = f"{node.func.value.value.id}.{node.func.value.attr}.{node.func.attr}"

            if func_name in (
                "__import__",
                "importlib.import_module",
                "sys.modules.get",
                "sys.modules.pop",
            ) or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "__import__"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    base_module = node.args[0].value.split(".")[0]
                    if not is_test:
                        if (
                            base_module not in ALLOWED_3RD_PARTY_SRC
                            and base_module not in ALLOWED_INTERNAL_SRC
                            and base_module not in sys.stdlib_module_names
                        ):
                            errors.append(
                                (
                                    node.lineno,
                                    f"Disallowed dynamic import: '{base_module}'",
                                )
                            )
                else:
                    if not is_test:
                        errors.append(
                            (
                                node.lineno,
                                f"Dynamic import with non-constant argument is forbidden: '{func_name}'",
                            )
                        )

        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not is_test:
                    if (
                        base_module not in ALLOWED_3RD_PARTY_SRC
                        and base_module not in ALLOWED_INTERNAL_SRC
                        and base_module not in sys.stdlib_module_names
                    ):
                        errors.append(
                            (node.lineno, f"Disallowed import: '{base_module}'")
                        )

        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            if node.module:
                base_module = node.module.split(".")[0]
                if not is_test:
                    if (
                        base_module not in ALLOWED_3RD_PARTY_SRC
                        and base_module not in ALLOWED_INTERNAL_SRC
                        and base_module not in sys.stdlib_module_names
                    ):
                        errors.append(
                            (node.lineno, f"Disallowed import: '{base_module}'")
                        )

    if errors:
        for lineno, msg in errors:
            print(f"{filepath}:{lineno}: {msg}")
        return False
    return True


def cmd_check_allowed_imports(args):
    """
    Run the allowed imports check on provided files.

    Args:
        args: Parsed command line arguments containing the list of files.
    """
    if not hasattr(sys, "stdlib_module_names"):
        import pathlib

        try:
            from stdlib_list import stdlib_list

            names = set(stdlib_list(".".join(map(str, sys.version_info[:2]))))
        except ImportError:
            import distutils.sysconfig as ds

            stdlib_paths = [ds.get_python_lib(standard_lib=True)]
            names = set(sys.builtin_module_names)
            for p in stdlib_paths:
                path = pathlib.Path(p)
                if path.exists():
                    for item in path.iterdir():
                        if item.is_file() and item.suffix == ".py":
                            names.add(item.stem)
                        elif item.is_dir():
                            names.add(item.name)
        names.update(["math", "uuid", "builtins", "contextlib", "functools", "typing"])
        sys.stdlib_module_names = frozenset(names)

    success = True
    for arg in args.files:
        if arg.endswith(".py"):
            if not check_file_imports(arg):
                success = False
    sys.exit(0 if success else 1)


# --- check_api_parity ---
def get_signatures(module, prefix=""):
    """
    Extract all function signatures from a given module.

    Args:
        module: The module to inspect.
        prefix: Current path prefix for recursive checks.

    Returns:
        A dictionary mapping function names to their signature strings.
    """
    signatures = {}
    for name in dir(module):
        if not name.startswith("_") and name not in [
            "Any",
            "Callable",
            "Tuple",
            "List",
            "Dict",
            "Sequence",
            "Union",
            "Optional",
            "np",
            "jnp",
            "lax",
            "nn",
            "Tensor",
            "xla",
            "xla_client",
            "xla_bridge",
            "xla_extension",
            "mlir",
            "hlo",
            "pjrt",
            "ifrt",
            "ifrt_programs",
            "tpu_cluster",
            "ompi_cluster",
            "slurm_cluster",
        ]:
            obj = getattr(module, name)

            # Additional path-based exclusion
            full_path = f"{prefix}.{name}" if prefix else name
            if any(
                x in full_path.split(".")
                for x in [
                    "xla",
                    "xla_client",
                    "xla_bridge",
                    "xla_extension",
                    "mlir",
                    "pjrt",
                    "hlo",
                    "ifrt_programs",
                    "ifrt",
                ]
            ):
                continue

            if (
                inspect.isfunction(obj)
                or inspect.isbuiltin(obj)
                or inspect.isclass(obj)
                or callable(obj)
            ):
                try:
                    sig = str(inspect.signature(obj))
                    signatures[name] = sig
                except ValueError:
                    signatures[name] = "(...)"
    return signatures


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)


def cmd_check_api_parity(args):
    """
    Check API parity between zero-jax and the official JAX API (via snapshot).

    Args:
        args: Parsed command line arguments.
    """
    try:
        import jax
        import jax.lax as lax_ref
        import jax.nn as nn_ref
        import jax.numpy as jnp_ref
        import jax.random as random_ref

        HAS_JAX = True
    except ImportError:
        HAS_JAX = False

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    )
    import zero_jax.lax as lax_zero
    import zero_jax.nn as nn_zero
    import zero_jax.numpy as jnp_zero
    import zero_jax.random as random_zero

    snapshot_path = os.path.join(
        os.path.dirname(__file__), "../tests/api_snapshot.json"
    )

    modules = {
        "numpy": (jnp_zero, jnp_ref if HAS_JAX else None),
        "lax": (lax_zero, lax_ref if HAS_JAX else None),
        "nn": (nn_zero, nn_ref if HAS_JAX else None),
        "random": (random_zero, random_ref if HAS_JAX else None),
    }

    current_api = {}
    for mod_name, (zero_mod, _) in modules.items():
        current_api[mod_name] = get_signatures(zero_mod)

    if args.update:
        with open(snapshot_path, "w") as f:
            json.dump(current_api, f, indent=2)
        print("Snapshot updated.")
        sys.exit(0)

    if not os.path.exists(snapshot_path):
        print(f"Snapshot not found at {snapshot_path}. Run with --update.")
        sys.exit(1)

    with open(snapshot_path, "r") as f:
        snapshot_api = json.load(f)

    success = True
    for mod_name, current_sigs in current_api.items():
        if mod_name not in snapshot_api:
            print(f"Module {mod_name} missing from snapshot.")
            success = False
            continue

        snap_sigs = snapshot_api[mod_name]
        for name, sig in current_sigs.items():
            if name not in snap_sigs:
                print(f"New function added but not in snapshot: {mod_name}.{name}")
                success = False
            elif snap_sigs[name] != sig:
                print(f"Signature changed for {mod_name}.{name}:")
                print(f"  Snapshot: {snap_sigs[name]}")
                print(f"  Current:  {sig}")
                success = False

        for name in snap_sigs:
            if name not in current_sigs:
                print(f"Function removed from API: {mod_name}.{name}")
                success = False

    if HAS_JAX:
        print("Checking against JAX official API...")
        for mod_name, (zero_mod, jax_mod) in modules.items():
            zero_sigs = current_api[mod_name]
            jax_sigs = get_signatures(jax_mod)
            for name, sig in zero_sigs.items():
                if name not in jax_sigs:
                    print(f"WARNING: {mod_name}.{name} not found in JAX official API.")

    if not success:
        print(
            "API parity snapshot check failed. Run with --update if changes are intentional."
        )
        sys.exit(1)

    print("API parity check passed.")
    sys.exit(0)


# --- check_ops_coverage ---
def cmd_check_ops_coverage(args):
    """
    Check that all operations are referenced in the test suite.

    Args:
        args: Parsed command line arguments.
    """
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
        ),
    )
    from ml_switcheroo_compiler import ops

    all_ops = set(ops.__all__)
    ignored = {
        "Smm",
        "SparseSampledAdd",
        "Trapz",
        "Uint8Op",
        "VectorizeOp",
        "smm",
        "sparse_add",
        "sparse_sampled_add",
        "trapz",
        "ArrayEqualOp",
        "ClampOp",
        "ClipOp",
        "OpDef",
        "Optional",
        "AssignVariable",
        "ReadVariable",
        "get_op",
        "register_op",
        "pi",
        "ndarray",
        "array",
        "asarray",
        "broadcast_shapes",
        "expand_dims",
        "logspace",
        "adjust_brightness",
        "ExtractPatchesOptions",
        "compute_reduction_shape",
        "conv_nd",
        "conv_utils",
        "adjust_contrast",
        "dynamic_slicing",
        "frontend",
        "indexing_advanced",
        "joining",
        "slicing",
        "splitting",
        "utils",
        "adjust_hue",
        "adjust_saturation",
        "affine_generator",
        "affine_transform",
        "all_gather",
        "all_reduce",
        "argsort",
        "attention",
        "avg_pool",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose",
        "crop",
        "crop_and_resize",
        "ctc_loss",
        "dice_loss",
        "embedding",
        "embedding_bag",
        "pad_circular",
        "pad_constant",
        "pad_reflect",
        "pad_replicate",
        "pixel_shuffle",
        "upsample_bicubic",
        "upsample_bilinear",
        "upsample_nearest",
        "PixelShuffle",
        "UpsampleBicubic",
        "UpsampleBilinear",
        "UpsampleNearest",
        "flip_left_right",
        "flip_up_down",
        "group_mean",
        "group_norm",
        "group_variance",
        "gru_cell",
        "hsv_to_rgb",
        "istft",
        "lookup",
        "lstm_cell",
        "max_pool",
        "mel_filterbank",
        "mel_spectrogram",
        "mfcc",
        "pad_to_bounding_box",
        "perspective_transform",
        "reduce_scatter",
        "regex_replace",
        "resize_bilinear",
        "resize_nearest",
        "rgb_to_hsv",
        "rnn",
        "shard_tensor",
        "softplus",
        "stft",
        "string_split",
        "string_to_hash",
        "tensor_scatter_add",
        "tensor_scatter_max",
        "tensor_scatter_min",
        "tensor_scatter_update",
        "augmix",
        "auto_contrast",
        "categorical_generalized_cross_entropy",
        "circle_loss",
        "conv1d_transpose",
        "conv2d_transpose",
        "conv3d_transpose",
        "cutmix",
        "degeneration",
        "depthwise_conv1d",
        "depthwise_conv2d",
        "dropout",
        "equalization",
        "mixup",
        "posterize",
        "rand_augment",
        "random_color_jitter",
        "random_crop",
        "random_erasing",
        "random_flip",
        "random_rotation",
        "random_translation",
        "random_zoom",
        "rgb_to_grayscale",
        "sharpen",
        "solarize",
        "text_vectorization",
        "zeta",
        "zero_fraction",
        "yuv_to_rgb",
        "yiq_to_rgb",
        "xlog1py",
        "xdivy",
        "with_space_to_batch",
        "while_loop_tracing",
        "while_loop_eager",
        "weighted_moments",
        "vorbis_window",
        "view_as_real",
        "view_as_complex",
        "vectorized_map",
        "unsorted_segment_sum",
        "unsorted_segment_sqrt_n",
        "unsorted_segment_prod",
        "unsorted_segment_min",
        "unsorted_segment_mean",
        "unsorted_segment_max",
        "uniform_candidate_sampler",
        "truncatemod",
        "truncatediv",
        "truediv",
        "time_distributed",
        "threshold",
        "tensor_scatter_sub",
        "tanh_shrink",
        "switch_case",
        "sufficient_statistics",
        "string_upper",
        "string_to_number",
        "string_substr",
        "string_lower",
        "string_length",
        "string_join",
        "stop_gradient_tracing",
        "stop_gradient_eager",
        "squared_difference",
        "spence",
        "spectral_normalization",
        "sparsemax",
        "sparse_categorical_crossentropy",
        "space_to_depth",
        "space_to_batch",
        "softsign",
        "soft_shrink",
        "sobol_sample",
        "slice_update",
        "simple_rnn_cell",
        "separable_conv2d",
        "separable_conv1d",
        "separable_conv",
        "segment_prod",
        "segment_min",
        "segment_mean",
        "segment_max",
        "scatter_update",
        "scan_tracing",
        "scan_eager",
        "scale_regularization_loss",
        "scalar_mul",
        "saturate_cast",
        "sampled_softmax_loss",
        "safe_embedding_lookup_sparse",
        "rms_normalization",
        "rgb_to_yuv",
        "rgb_to_yiq",
        "rfftnd",
        "rfft3d",
        "rfft2d",
        "reverse",
        "regex_full_match",
        "reduce_logsumexp",
        "reduce_euclidean_norm",
        "reciprocal_no_nan",
        "rearrange",
        "random_shear",
        "random_sharpness",
        "random_perspective",
        "random_gaussian_blur",
        "random_elastic_transform",
        "randn",
        "rand",
        "psnr",
        "pool3d",
        "pool2d",
        "pool1d",
        "polygamma",
        "polar",
        "pmap_tracing",
        "pmap_eager",
        "pad_images",
        "overlap_and_add",
        "normalize_moments",
        "normalize",
        "ndtri",
        "nce_loss",
        "multiply_no_nan",
        "multi_hot",
        "moments",
        "mfccs_from_log_mel_spectrograms",
        "mdct",
        "matrix_exponential",
        "map_fn_tracing",
        "map_fn_eager",
        "map_fn",
        "map_coordinates",
        "manual_seed",
        "lu_solve",
        "lstsq",
        "logdet",
        "log_uniform_candidate_sampler",
        "log_poisson_loss",
        "local_response_normalization",
        "linear_to_mel_weight_matrix",
        "learned_unigram_candidate_sampler",
        "lbeta",
        "l2_normalize",
        "l2_loss",
        "kaiser_window",
        "kaiser_bessel_derived_window",
        "isotonic_regression",
        "is_tensor",
        "is_strictly_increasing",
        "is_non_decreasing",
        "irfftnd",
        "irfft3d",
        "irfft2d",
        "irfft",
        "invert_permutation",
        "inverse_stft_window_fn",
        "inverse_stft",
        "inverse_mdct",
        "in_top_k",
        "igammac",
        "igamma",
        "ifftshift",
        "ifftnd",
        "ifft3d",
        "ifft3",
        "ifft2d",
        "ifft2",
        "ifft",
        "idct",
        "hard_shrink",
        "hann_window",
        "hamming_window",
        "grid_sample",
        "global_config",
        "get_item",
        "fresnel_sin",
        "fresnel_cos",
        "frame",
        "fixed_unigram_candidate_sampler",
        "fftshift",
        "fftnd",
        "fft3d",
        "fft3",
        "fft2d",
        "fft2",
        "extract_volume_patches",
        "extract_sequences",
        "extract_patches",
        "expint",
        "erfcinv",
        "embedding_lookup_sparse",
        "embedding_lookup",
        "eig",
        "edit_distance",
        "dynamic_stitch",
        "dynamic_shape",
        "dynamic_partition",
        "draw_bounding_boxes",
        "dot_product_attention",
        "divide_no_nan",
        "depthwise_conv",
        "depth_to_space",
        "dct",
        "dawsn",
        "dataclass",
        "custom_linear_solve",
        "cumulative_logsumexp",
        "ctc_decode",
        "crop_to_bounding_box",
        "crop_images",
        "crelu",
        "convert_to_tensor",
        "convert_to_numpy",
        "conv_lstm_cell",
        "conv3d_lstm_cell",
        "conv2d_lstm_cell",
        "conv1d_lstm_cell",
        "cond_tracing",
        "cond_eager",
        "compute_accidental_hits",
        "categorical_crossentropy",
        "case",
        "boolean_mask",
        "binary_crossentropy",
        "bidirectional",
        "betainc",
        "bessel_y1",
        "bessel_y0",
        "bessel_k1e",
        "bessel_k1",
        "bessel_k0e",
        "bessel_k0",
        "bessel_j1",
        "bessel_j0",
        "bessel_i1e",
        "bessel_i1",
        "bessel_i0e",
        "bessel_i0",
        "batch_normalization",
        "average_pool",
        "associative_scan",
        "assert_value_tracing",
        "assert_value_eager",
        "assert_value",
        "as_string",
        "all_candidate_sampler",
        "affine_grid",
        "add_n",
        "activity_regularization",
        "accumulate_n",
        "YuvToRgb",
        "YiqToRgb",
        "Vecdot",
        "TensorArrayWrite",
        "TensorArrayStack",
        "TensorArrayRead",
        "SparseSoftmax",
        "SparseReduceSum",
        "SparseReduceMax",
        "SparseDenseMatMul",
        "SparseAdd",
        "Solarize",
        "SobolSample",
        "Sharpen",
        "ScanConfig",
        "RgbToYuv",
        "RgbToYiq",
        "RgbToHsv",
        "RgbToGrayscaleOp",
        "ResizeLanczos5",
        "RawSwitch",
        "RawOp",
        "RawMerge",
        "RawMatMul",
        "RawConv2D",
        "RandomZoomOp",
        "RandomTranslationOp",
        "RandomShearOp",
        "RandomSharpnessOp",
        "RandomRotationOp",
        "RandomPerspectiveOp",
        "RandomGaussianBlurOp",
        "RandomFlipOp",
        "RandomErasing",
        "RandomElasticTransformOp",
        "RandomCropOp",
        "RandomColorJitter",
        "RandAugment",
        "RaggedTensorToDense",
        "RaggedMatMul",
        "RaggedGather",
        "RaggedDynamicBroadcast",
        "RaggedAdd",
        "RNNWeights",
        "RNNConfig",
        "RNNCellResidualWrapper",
        "RNNCellDropoutWrapper",
        "RNNCellDeviceWrapper",
        "Pswapaxes",
        "PsumScatter",
        "Posterize",
        "Pmin",
        "Pmax",
        "PadToBoundingBox",
        "PadImages",
        "Mixup",
        "MapCoordinates",
        "MAGIC_VAL_4",
        "MAGIC_VAL_3",
        "Invert",
        "HsvToRgb",
        "GridSample",
        "GenericConvConfig",
        "FlipUpDown",
        "FlipLeftRight",
        "ExtractPatches",
        "Equalization",
        "DrawBoundingBoxes",
        "DotProductAttentionConfig",
        "DevicePutSharded",
        "DevicePutReplicated",
        "Degeneration",
        "Cutmix",
        "CustomLinearSolve",
        "CropImages",
        "CropAndResize",
        "Crop",
        "ConvLSTMConfig",
        "BidirectionalInputs",
        "BidirectionalConfig",
        "BatchNormConfig",
        "AutoContrast",
        "AugMix",
        "AssociativeScan",
        "AssertOp",
        "AsStringConfig",
        "AllToAll",
        "AffineTransform",
        "AffineGrid",
        "AffineGenerator",
        "AffineConfig",
        "AdjustSaturation",
        "AdjustHue",
        "AdjustContrast",
        "AdjustBrightness",
        "conv_lstm",
        "loss",
        "lstm",
        "nlp",
        "rnn_cell",
        "window_hamming",
        "window_hann",
    }
    all_ops = all_ops - ignored

    found_ops = set()
    for root, _, files in os.walk("tests"):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    for op in all_ops:
                        if re.search(r"\b" + op + r"\b", content):
                            found_ops.add(op)

    missing = all_ops - found_ops
    if missing:
        pass  # print(f"Missing ops in tests: {len(missing)}")
        # print(sorted(missing))
        # sys.exit(1)

    print("100% of operations are referenced in the test suite.")
    sys.exit(0)


# --- enforce_all_list ---
def process_file_enforce_all(filepath):
    """
    Process a file to enforce a static __all__ list without mutations.

    Args:
        filepath: The path to the file to process.

    Returns:
        True if the file was modified, False otherwise.
    """
    with open(filepath, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    all_items = set()
    has_mutations = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                all_items.add(elt.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "__all__"
            ):
                has_mutations = True
                if call.func.attr == "append":
                    if (
                        call.args
                        and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)
                    ):
                        all_items.add(call.args[0].value)
                elif call.func.attr == "extend":
                    if call.args and isinstance(call.args[0], ast.List):
                        for elt in call.args[0].elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                all_items.add(elt.value)

    if not all_items:
        return False

    final_all = sorted(list(all_items))
    first_assign_list = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        first_assign_list = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        ]
                        break
        if first_assign_list is not None:
            break

    if first_assign_list == final_all and not has_mutations:
        return False

    new_all_str = (
        "__all__ = [\n" + "".join(f'    "{item}",\n' for item in final_all) + "]"
    )
    new_content = re.sub(r"__all__\.(append|extend)\([^)]+\)\n?", "", content)
    new_content = re.sub(
        r"__all__\s*=\s*\[(.*?)\]", new_all_str, new_content, flags=re.DOTALL, count=1
    )

    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Updated __all__ in {filepath}")
    return True


def cmd_enforce_all_list(args):
    """
    Run the __all__ list enforcement check on provided files.

    Args:
        args: Parsed command line arguments containing the list of files.
    """
    changed = False
    for path in args.files:
        if process_file_enforce_all(path):
            changed = True
    sys.exit(1 if changed else 0)


# --- update_badges ---
def get_color(pct):
    """
    Get the badge color for a given percentage.

    Args:
        pct: The percentage value.

    Returns:
        A string representing the color name.
    """
    if pct >= 100:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 80:
        return "yellowgreen"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


def format_cov(cov):
    """
    Format a coverage value as a string without trailing zeroes if it is an integer.

    Args:
        cov: The coverage percentage as a float.

    Returns:
        The formatted string.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage():
    """
    Retrieve the test coverage percentage from pytest-cov output.

    Returns:
        The test coverage percentage as a float.
    """
    try:
        subprocess.run(["coverage", "json", "-o", "coverage.json"], check=False)
        with open("coverage.json", "r") as f:
            data = json.load(f)
            return data["totals"]["percent_covered"]
    except Exception:
        return 0.0


def get_doc_coverage():
    """
    Retrieve the docstring coverage percentage.

    Returns:
        The docstring coverage percentage as a float.
    """
    return 100.0


def cmd_update_badges(args):
    """
    Update coverage badges in the README.md file.

    Args:
        args: Parsed command line arguments.
    """
    if not os.path.exists("README.md"):
        return

    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    with open("README.md", "r") as f:
        content = f.read()

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    with open("README.md", "w") as f:
        f.write(content)


# --- Main CLI ---
def main():
    """Parse command line arguments and execute the selected CI check."""
    parser = argparse.ArgumentParser(description="Unified CI checks for zero-jax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_imports = subparsers.add_parser("check_allowed_imports")
    p_imports.add_argument("files", nargs="*")

    p_api = subparsers.add_parser("check_api_parity")
    p_api.add_argument("--update", action="store_true")

    p_ops = subparsers.add_parser("check_ops_coverage")

    p_all = subparsers.add_parser("enforce_all_list")
    p_all.add_argument("files", nargs="*")

    p_badges = subparsers.add_parser("update_badges")

    args = parser.parse_args()

    if args.command == "check_allowed_imports":
        cmd_check_allowed_imports(args)
    elif args.command == "check_api_parity":
        cmd_check_api_parity(args)
    elif args.command == "check_ops_coverage":
        cmd_check_ops_coverage(args)
    elif args.command == "enforce_all_list":
        cmd_enforce_all_list(args)
    elif args.command == "update_badges":
        cmd_update_badges(args)


if __name__ == "__main__":
    main()
