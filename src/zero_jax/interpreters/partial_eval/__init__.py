"""Frontend API routing for jax.interpreters.partial_eval."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def AbstractedAxesSpec(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AbstractedAxesSpec."""
    return getattr(_ops, "AbstractedAxesSpec")(*args, **kwargs)


class AbstractedAxisName:
    """Mock implementation for AbstractedAxisName."""

    pass


class BoundedAxisSize:
    """BoundedAxisSize(val, bound)"""

    pass


def Const(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "Const")(*args, **kwargs)


def ConstFoldRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ConstFoldRule."""
    return getattr(_ops, "ConstFoldRule")(*args, **kwargs)


class ConstVar:
    """ConstVar(val,)"""

    pass


def DCERule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DCERule."""
    return getattr(_ops, "DCERule")(*args, **kwargs)


class DebugInfo:
    """DebugInfo(func_src_info, signature, in_tree, out_tree, has_kwargs, traced_for)"""

    pass


class DynamicJaxprTrace:
    """Mock implementation for DynamicJaxprTrace."""

    pass


class DynamicJaxprTracer:
    """Mock implementation for DynamicJaxprTracer."""

    pass


def ForwardingRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ForwardingRule."""
    return getattr(_ops, "ForwardingRule")(*args, **kwargs)


class FreeVar:
    """FreeVar(val,)"""

    pass


class Jaxpr:
    """Mock implementation for Jaxpr."""

    pass


class JaxprEqnRecipe:
    """JaxprEqnRecipe(eqn_id, in_tracers, out_tracer_refs, out_avals, primitive, params, effects, source_info, ctx)"""

    pass


class JaxprStackFrame:
    """Mock implementation for JaxprStackFrame."""

    pass


def JaxprTracerRecipe(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for JaxprTracerRecipe."""
    return getattr(_ops, "JaxprTracerRecipe")(*args, **kwargs)


class LambdaBinding:
    """LambdaBinding()"""

    pass


def ParamsUpdater(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ParamsUpdater."""
    return getattr(_ops, "ParamsUpdater")(*args, **kwargs)


class PartialEvalCustomResult:
    """Built-in immutable sequence."""

    pass


def PartialEvalCustomRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for PartialEvalCustomRule."""
    return getattr(_ops, "PartialEvalCustomRule")(*args, **kwargs)


class PartialVal:
    """Partial value: either a known value or an unknown (abstract) value."""

    pass


def ResAvalUpdater(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ResAvalUpdater."""
    return getattr(_ops, "ResAvalUpdater")(*args, **kwargs)


class TracerAsName:
    """Mock implementation for TracerAsName."""

    pass


class TracerId:
    """int([x]) -> integer"""

    pass


def abstract_eval_fun(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for abstract_eval_fun."""
    return getattr(_ops, "abstract_eval_fun")(*args, **kwargs)


def arg_info_all(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for arg_info_all."""
    return getattr(_ops, "arg_info_all")(*args, **kwargs)


def call_padding_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_padding_rule."""
    return getattr(_ops, "call_padding_rule")(*args, **kwargs)


call_param_updaters: Any = None


def call_partial_eval_custom_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_partial_eval_custom_rule."""
    return getattr(_ops, "call_partial_eval_custom_rule")(*args, **kwargs)


call_partial_eval_rules: Any = None


def close_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for close_jaxpr."""
    return getattr(_ops, "close_jaxpr")(*args, **kwargs)


def closed_call_partial_eval_custom_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for closed_call_partial_eval_custom_rule."""
    return getattr(_ops, "closed_call_partial_eval_custom_rule")(*args, **kwargs)


from . import config

const_fold_rules: Any = None


def convert_constvars_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_constvars_jaxpr."""
    return getattr(_ops, "convert_constvars_jaxpr")(*args, **kwargs)


def convert_envvars_to_constvars(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_envvars_to_constvars."""
    return getattr(_ops, "convert_envvars_to_constvars")(*args, **kwargs)


def convert_invars_to_constvars(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_invars_to_constvars."""
    return getattr(_ops, "convert_invars_to_constvars")(*args, **kwargs)


custom_partial_eval_rules: Any = None

custom_staging_rules: Any = None


def dce_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr."""
    return getattr(_ops, "dce_jaxpr")(*args, **kwargs)


def dce_jaxpr_call_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_call_rule."""
    return getattr(_ops, "dce_jaxpr_call_rule")(*args, **kwargs)


def dce_jaxpr_closed_call_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_closed_call_rule."""
    return getattr(_ops, "dce_jaxpr_closed_call_rule")(*args, **kwargs)


def dce_jaxpr_consts(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_consts."""
    return getattr(_ops, "dce_jaxpr_consts")(*args, **kwargs)


dce_rules: Any = None


def debug_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for debug_info."""
    return getattr(_ops, "debug_info")(*args, **kwargs)


def debug_info_final(*args: Any, **kwargs: Any) -> Any:
    """Make a DebugInfo from data available to final-style primitives like pmap."""
    return getattr(_ops, "debug_info_final")(*args, **kwargs)


def def_trivial_padding(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for def_trivial_padding."""
    return getattr(_ops, "def_trivial_padding")(*args, **kwargs)


def extend_jaxpr_stack(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_jaxpr_stack."""
    return getattr(_ops, "extend_jaxpr_stack")(*args, **kwargs)


forwarding_rules: Any = None


def infer_lambda_input_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for infer_lambda_input_type."""
    return getattr(_ops, "infer_lambda_input_type")(*args, **kwargs)


def instantiate_const_at(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for instantiate_const_at."""
    return getattr(_ops, "instantiate_const_at")(*args, **kwargs)


def make_jaxpr_effects(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_jaxpr_effects."""
    return getattr(_ops, "make_jaxpr_effects")(*args, **kwargs)


def move_binders_to_back(*args: Any, **kwargs: Any) -> Any:
    """Reorder `invars` by moving those indicated in `to_move` to the back."""
    return getattr(_ops, "move_binders_to_back")(*args, **kwargs)


def move_binders_to_front(*args: Any, **kwargs: Any) -> Any:
    """Reorder `invars` by moving those indicated in `to_move` to the front."""
    return getattr(_ops, "move_binders_to_front")(*args, **kwargs)


def new_eqn_recipe(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_eqn_recipe."""
    return getattr(_ops, "new_eqn_recipe")(*args, **kwargs)


def pad_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pad_jaxpr."""
    return getattr(_ops, "pad_jaxpr")(*args, **kwargs)


padding_rules: Any = None


def partial_eval_jaxpr_custom(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partial_eval_jaxpr_custom."""
    return getattr(_ops, "partial_eval_jaxpr_custom")(*args, **kwargs)


def partial_eval_jaxpr_custom_rule_not_implemented(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partial_eval_jaxpr_custom_rule_not_implemented."""
    return getattr(_ops, "partial_eval_jaxpr_custom_rule_not_implemented")(
        *args, **kwargs
    )


partial_eval_jaxpr_custom_rules: Any = None


def partial_eval_jaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """Unzip a jaxpr in two by data dependence into 'known' and 'unknown' parts."""
    return getattr(_ops, "partial_eval_jaxpr_nounits")(*args, **kwargs)


def partial_eval_wrapper_nounits(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "partial_eval_wrapper_nounits")(*args, **kwargs)


def partition_pvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partition_pvals."""
    return getattr(_ops, "partition_pvals")(*args, **kwargs)


def recipe_to_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for recipe_to_eqn."""
    return getattr(_ops, "recipe_to_eqn")(*args, **kwargs)


def result_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for result_info."""
    return getattr(_ops, "result_info")(*args, **kwargs)


def sig_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for sig_info."""
    return getattr(_ops, "sig_info")(*args, **kwargs)


def trace_to_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Partially evaluate a function, building a jaxpr for un-evaluated computation."""
    return getattr(_ops, "trace_to_jaxpr")(*args, **kwargs)


def trace_to_jaxpr_dynamic(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_dynamic."""
    return getattr(_ops, "trace_to_jaxpr_dynamic")(*args, **kwargs)


def trace_to_jaxpr_dynamic2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_dynamic2."""
    return getattr(_ops, "trace_to_jaxpr_dynamic2")(*args, **kwargs)


def trace_to_jaxpr_final(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_final."""
    return getattr(_ops, "trace_to_jaxpr_final")(*args, **kwargs)


def trace_to_jaxpr_final2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_final2."""
    return getattr(_ops, "trace_to_jaxpr_final2")(*args, **kwargs)


def trace_to_jaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_nounits."""
    return getattr(_ops, "trace_to_jaxpr_nounits")(*args, **kwargs)


def trace_to_subjaxpr(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "trace_to_subjaxpr")(*args, **kwargs)


def trace_to_subjaxpr_dynamic(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_subjaxpr_dynamic."""
    return getattr(_ops, "trace_to_subjaxpr_dynamic")(*args, **kwargs)


def trace_to_subjaxpr_dynamic2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_subjaxpr_dynamic2."""
    return getattr(_ops, "trace_to_subjaxpr_dynamic2")(*args, **kwargs)


def trace_to_subjaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "trace_to_subjaxpr_nounits")(*args, **kwargs)


def trace_to_subjaxpr_nounits_dyn(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "trace_to_subjaxpr_nounits_dyn")(*args, **kwargs)


def trace_to_subjaxpr_nounits_fwd(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "trace_to_subjaxpr_nounits_fwd")(*args, **kwargs)


def tracers_to_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Constructs Jaxpr given tracers for inputs and outputs."""
    return getattr(_ops, "tracers_to_jaxpr")(*args, **kwargs)


def trivial_ctx(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trivial_ctx."""
    return getattr(_ops, "trivial_ctx")(*args, **kwargs)
