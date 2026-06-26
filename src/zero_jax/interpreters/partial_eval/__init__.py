"""Frontend API routing for jax.interpreters.partial_eval."""

from typing import Any


def AbstractedAxesSpec(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AbstractedAxesSpec."""
    raise NotImplementedError("AbstractedAxesSpec not yet implemented in zero-jax")


class AbstractedAxisName:
    """Mock implementation for AbstractedAxisName."""

    pass


class BoundedAxisSize:
    """BoundedAxisSize(val, bound)"""

    pass


def Const(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("Const not yet implemented in zero-jax")


def ConstFoldRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ConstFoldRule."""
    raise NotImplementedError("ConstFoldRule not yet implemented in zero-jax")


class ConstVar:
    """ConstVar(val,)"""

    pass


def DCERule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for DCERule."""
    raise NotImplementedError("DCERule not yet implemented in zero-jax")


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
    raise NotImplementedError("ForwardingRule not yet implemented in zero-jax")


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
    raise NotImplementedError("JaxprTracerRecipe not yet implemented in zero-jax")


class LambdaBinding:
    """LambdaBinding()"""

    pass


def ParamsUpdater(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ParamsUpdater."""
    raise NotImplementedError("ParamsUpdater not yet implemented in zero-jax")


class PartialEvalCustomResult:
    """Built-in immutable sequence."""

    pass


def PartialEvalCustomRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for PartialEvalCustomRule."""
    raise NotImplementedError("PartialEvalCustomRule not yet implemented in zero-jax")


class PartialVal:
    """Partial value: either a known value or an unknown (abstract) value."""

    pass


def ResAvalUpdater(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ResAvalUpdater."""
    raise NotImplementedError("ResAvalUpdater not yet implemented in zero-jax")


class TracerAsName:
    """Mock implementation for TracerAsName."""

    pass


class TracerId:
    """int([x]) -> integer"""

    pass


def abstract_eval_fun(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for abstract_eval_fun."""
    raise NotImplementedError("abstract_eval_fun not yet implemented in zero-jax")


def arg_info_all(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for arg_info_all."""
    raise NotImplementedError("arg_info_all not yet implemented in zero-jax")


def call_padding_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_padding_rule."""
    raise NotImplementedError("call_padding_rule not yet implemented in zero-jax")


call_param_updaters: Any = None


def call_partial_eval_custom_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_partial_eval_custom_rule."""
    raise NotImplementedError(
        "call_partial_eval_custom_rule not yet implemented in zero-jax"
    )


call_partial_eval_rules: Any = None


def close_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for close_jaxpr."""
    raise NotImplementedError("close_jaxpr not yet implemented in zero-jax")


def closed_call_partial_eval_custom_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for closed_call_partial_eval_custom_rule."""
    raise NotImplementedError(
        "closed_call_partial_eval_custom_rule not yet implemented in zero-jax"
    )


from . import config

const_fold_rules: Any = None


def convert_constvars_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_constvars_jaxpr."""
    raise NotImplementedError("convert_constvars_jaxpr not yet implemented in zero-jax")


def convert_envvars_to_constvars(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_envvars_to_constvars."""
    raise NotImplementedError(
        "convert_envvars_to_constvars not yet implemented in zero-jax"
    )


def convert_invars_to_constvars(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for convert_invars_to_constvars."""
    raise NotImplementedError(
        "convert_invars_to_constvars not yet implemented in zero-jax"
    )


custom_partial_eval_rules: Any = None

custom_staging_rules: Any = None


def dce_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr."""
    raise NotImplementedError("dce_jaxpr not yet implemented in zero-jax")


def dce_jaxpr_call_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_call_rule."""
    raise NotImplementedError("dce_jaxpr_call_rule not yet implemented in zero-jax")


def dce_jaxpr_closed_call_rule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_closed_call_rule."""
    raise NotImplementedError(
        "dce_jaxpr_closed_call_rule not yet implemented in zero-jax"
    )


def dce_jaxpr_consts(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dce_jaxpr_consts."""
    raise NotImplementedError("dce_jaxpr_consts not yet implemented in zero-jax")


dce_rules: Any = None


def debug_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for debug_info."""
    raise NotImplementedError("debug_info not yet implemented in zero-jax")


def debug_info_final(*args: Any, **kwargs: Any) -> Any:
    """Make a DebugInfo from data available to final-style primitives like pmap."""
    raise NotImplementedError("debug_info_final not yet implemented in zero-jax")


def def_trivial_padding(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for def_trivial_padding."""
    raise NotImplementedError("def_trivial_padding not yet implemented in zero-jax")


def extend_jaxpr_stack(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_jaxpr_stack."""
    raise NotImplementedError("extend_jaxpr_stack not yet implemented in zero-jax")


forwarding_rules: Any = None


def infer_lambda_input_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for infer_lambda_input_type."""
    raise NotImplementedError("infer_lambda_input_type not yet implemented in zero-jax")


def instantiate_const_at(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for instantiate_const_at."""
    raise NotImplementedError("instantiate_const_at not yet implemented in zero-jax")


def make_jaxpr_effects(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_jaxpr_effects."""
    raise NotImplementedError("make_jaxpr_effects not yet implemented in zero-jax")


def move_binders_to_back(*args: Any, **kwargs: Any) -> Any:
    """Reorder `invars` by moving those indicated in `to_move` to the back."""
    raise NotImplementedError("move_binders_to_back not yet implemented in zero-jax")


def move_binders_to_front(*args: Any, **kwargs: Any) -> Any:
    """Reorder `invars` by moving those indicated in `to_move` to the front."""
    raise NotImplementedError("move_binders_to_front not yet implemented in zero-jax")


def new_eqn_recipe(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_eqn_recipe."""
    raise NotImplementedError("new_eqn_recipe not yet implemented in zero-jax")


def pad_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pad_jaxpr."""
    raise NotImplementedError("pad_jaxpr not yet implemented in zero-jax")


padding_rules: Any = None


def partial_eval_jaxpr_custom(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partial_eval_jaxpr_custom."""
    raise NotImplementedError(
        "partial_eval_jaxpr_custom not yet implemented in zero-jax"
    )


def partial_eval_jaxpr_custom_rule_not_implemented(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partial_eval_jaxpr_custom_rule_not_implemented."""
    raise NotImplementedError(
        "partial_eval_jaxpr_custom_rule_not_implemented not yet implemented in zero-jax"
    )


partial_eval_jaxpr_custom_rules: Any = None


def partial_eval_jaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """Unzip a jaxpr in two by data dependence into 'known' and 'unknown' parts."""
    raise NotImplementedError(
        "partial_eval_jaxpr_nounits not yet implemented in zero-jax"
    )


def partial_eval_wrapper_nounits(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError(
        "partial_eval_wrapper_nounits not yet implemented in zero-jax"
    )


def partition_pvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for partition_pvals."""
    raise NotImplementedError("partition_pvals not yet implemented in zero-jax")


def recipe_to_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for recipe_to_eqn."""
    raise NotImplementedError("recipe_to_eqn not yet implemented in zero-jax")


def result_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for result_info."""
    raise NotImplementedError("result_info not yet implemented in zero-jax")


def sig_info(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for sig_info."""
    raise NotImplementedError("sig_info not yet implemented in zero-jax")


def trace_to_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Partially evaluate a function, building a jaxpr for un-evaluated computation."""
    raise NotImplementedError("trace_to_jaxpr not yet implemented in zero-jax")


def trace_to_jaxpr_dynamic(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_dynamic."""
    raise NotImplementedError("trace_to_jaxpr_dynamic not yet implemented in zero-jax")


def trace_to_jaxpr_dynamic2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_dynamic2."""
    raise NotImplementedError("trace_to_jaxpr_dynamic2 not yet implemented in zero-jax")


def trace_to_jaxpr_final(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_final."""
    raise NotImplementedError("trace_to_jaxpr_final not yet implemented in zero-jax")


def trace_to_jaxpr_final2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_final2."""
    raise NotImplementedError("trace_to_jaxpr_final2 not yet implemented in zero-jax")


def trace_to_jaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_jaxpr_nounits."""
    raise NotImplementedError("trace_to_jaxpr_nounits not yet implemented in zero-jax")


def trace_to_subjaxpr(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("trace_to_subjaxpr not yet implemented in zero-jax")


def trace_to_subjaxpr_dynamic(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_subjaxpr_dynamic."""
    raise NotImplementedError(
        "trace_to_subjaxpr_dynamic not yet implemented in zero-jax"
    )


def trace_to_subjaxpr_dynamic2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_to_subjaxpr_dynamic2."""
    raise NotImplementedError(
        "trace_to_subjaxpr_dynamic2 not yet implemented in zero-jax"
    )


def trace_to_subjaxpr_nounits(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError(
        "trace_to_subjaxpr_nounits not yet implemented in zero-jax"
    )


def trace_to_subjaxpr_nounits_dyn(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError(
        "trace_to_subjaxpr_nounits_dyn not yet implemented in zero-jax"
    )


def trace_to_subjaxpr_nounits_fwd(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError(
        "trace_to_subjaxpr_nounits_fwd not yet implemented in zero-jax"
    )


def tracers_to_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Constructs Jaxpr given tracers for inputs and outputs."""
    raise NotImplementedError("tracers_to_jaxpr not yet implemented in zero-jax")


def trivial_ctx(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trivial_ctx."""
    raise NotImplementedError("trivial_ctx not yet implemented in zero-jax")
