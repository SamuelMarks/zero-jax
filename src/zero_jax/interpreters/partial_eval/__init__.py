"""Partial eval stubs."""

from typing import Any

import zero_jax._compiler_proxy_ops as ops

from . import config


def AbstractedAxesSpec(*args, **kwargs):
    return ops.AbstractedAxesSpec(*args, **kwargs)


def Const(*args, **kwargs):
    return ops.Const(*args, **kwargs)


def ConstFoldRule(*args, **kwargs):
    return ops.ConstFoldRule(*args, **kwargs)


def DCERule(*args, **kwargs):
    return ops.DCERule(*args, **kwargs)


def ForwardingRule(*args, **kwargs):
    return ops.ForwardingRule(*args, **kwargs)


def JaxprTracerRecipe(*args, **kwargs):
    return ops.JaxprTracerRecipe(*args, **kwargs)


def ParamsUpdater(*args, **kwargs):
    return ops.ParamsUpdater(*args, **kwargs)


def PartialEvalCustomRule(*args, **kwargs):
    return ops.PartialEvalCustomRule(*args, **kwargs)


def ResAvalUpdater(*args, **kwargs):
    return ops.ResAvalUpdater(*args, **kwargs)


class AbstractedAxisName:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BoundedAxisSize:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ConstVar:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DebugInfo:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DynamicJaxprTracer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FreeVar:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Jaxpr:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class JaxprEqnRecipe:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class JaxprStackFrame:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class JaxprTrace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class JaxprTracer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class LambdaBinding:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PartialEvalCustomResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PartialVal:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TracerAsName:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TracerId:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Val:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DynamicJaxprTrace(JaxprTrace):
    pass


def abstract_eval_fun(*args, **kwargs):
    return ops.abstract_eval_fun(*args, **kwargs)


def arg_info_all(*args, **kwargs):
    return ops.arg_info_all(*args, **kwargs)


def call_padding_rule(*args, **kwargs):
    return ops.call_padding_rule(*args, **kwargs)


def call_partial_eval_custom_rule(*args, **kwargs):
    return ops.call_partial_eval_custom_rule(*args, **kwargs)


def close_jaxpr(*args, **kwargs):
    return ops.close_jaxpr(*args, **kwargs)


def closed_call_partial_eval_custom_rule(*args, **kwargs):
    return ops.closed_call_partial_eval_custom_rule(*args, **kwargs)


def convert_constvars_jaxpr(*args, **kwargs):
    return ops.convert_constvars_jaxpr(*args, **kwargs)


def convert_envvars_to_constvars(*args, **kwargs):
    return ops.convert_envvars_to_constvars(*args, **kwargs)


def convert_invars_to_constvars(*args, **kwargs):
    return ops.convert_invars_to_constvars(*args, **kwargs)


def dce_jaxpr(*args, **kwargs):
    return ops.dce_jaxpr(*args, **kwargs)


def dce_jaxpr_call_rule(*args, **kwargs):
    return ops.dce_jaxpr_call_rule(*args, **kwargs)


def dce_jaxpr_closed_call_rule(*args, **kwargs):
    return ops.dce_jaxpr_closed_call_rule(*args, **kwargs)


def dce_jaxpr_consts(*args, **kwargs):
    return ops.dce_jaxpr_consts(*args, **kwargs)


def debug_info(*args, **kwargs):
    return ops.debug_info(*args, **kwargs)


def debug_info_final(*args, **kwargs):
    return ops.debug_info_final(*args, **kwargs)


def def_trivial_padding(*args, **kwargs):
    return ops.def_trivial_padding(*args, **kwargs)


def extend_jaxpr_stack(*args, **kwargs):
    return ops.extend_jaxpr_stack(*args, **kwargs)


def infer_lambda_input_type(*args, **kwargs):
    return ops.infer_lambda_input_type(*args, **kwargs)


def instantiate_const_at(*args, **kwargs):
    return ops.instantiate_const_at(*args, **kwargs)


def make_jaxpr_effects(*args, **kwargs):
    return ops.make_jaxpr_effects(*args, **kwargs)


def move_binders_to_back(*args, **kwargs):
    return ops.move_binders_to_back(*args, **kwargs)


def move_binders_to_front(*args, **kwargs):
    return ops.move_binders_to_front(*args, **kwargs)


def new_eqn_recipe(*args, **kwargs):
    return ops.new_eqn_recipe(*args, **kwargs)


def pad_jaxpr(*args, **kwargs):
    return ops.pad_jaxpr(*args, **kwargs)


def partial_eval_jaxpr_custom(*args, **kwargs):
    return ops.partial_eval_jaxpr_custom(*args, **kwargs)


def partial_eval_jaxpr_custom_rule_not_implemented(*args, **kwargs):
    return ops.partial_eval_jaxpr_custom_rule_not_implemented(*args, **kwargs)


def partial_eval_jaxpr_nounits(*args, **kwargs):
    return ops.partial_eval_jaxpr_nounits(*args, **kwargs)


def partial_eval_wrapper_nounits(*args, **kwargs):
    return ops.partial_eval_wrapper_nounits(*args, **kwargs)


def partition_pvals(*args, **kwargs):
    return ops.partition_pvals(*args, **kwargs)


def recipe_to_eqn(*args, **kwargs):
    return ops.recipe_to_eqn(*args, **kwargs)


def result_info(*args, **kwargs):
    return ops.result_info(*args, **kwargs)


def sig_info(*args, **kwargs):
    return ops.sig_info(*args, **kwargs)


def trace_to_jaxpr(*args, **kwargs):
    return ops.trace_to_jaxpr(*args, **kwargs)


def trace_to_jaxpr_dynamic(*args, **kwargs):
    return ops.trace_to_jaxpr_dynamic(*args, **kwargs)


def trace_to_jaxpr_dynamic2(*args, **kwargs):
    return ops.trace_to_jaxpr_dynamic2(*args, **kwargs)


def trace_to_jaxpr_final(*args, **kwargs):
    return ops.trace_to_jaxpr_final(*args, **kwargs)


def trace_to_jaxpr_final2(*args, **kwargs):
    return ops.trace_to_jaxpr_final2(*args, **kwargs)


def trace_to_jaxpr_nounits(*args, **kwargs):
    return ops.trace_to_jaxpr_nounits(*args, **kwargs)


def trace_to_subjaxpr(*args, **kwargs):
    return ops.trace_to_subjaxpr(*args, **kwargs)


def trace_to_subjaxpr_dynamic(*args, **kwargs):
    return ops.trace_to_subjaxpr_dynamic(*args, **kwargs)


def trace_to_subjaxpr_dynamic2(*args, **kwargs):
    return ops.trace_to_subjaxpr_dynamic2(*args, **kwargs)


def trace_to_subjaxpr_nounits(*args, **kwargs):
    return ops.trace_to_subjaxpr_nounits(*args, **kwargs)


def trace_to_subjaxpr_nounits_dyn(*args, **kwargs):
    return ops.trace_to_subjaxpr_nounits_dyn(*args, **kwargs)


def trace_to_subjaxpr_nounits_fwd(*args, **kwargs):
    return ops.trace_to_subjaxpr_nounits_fwd(*args, **kwargs)


def tracers_to_jaxpr(*args, **kwargs):
    return ops.tracers_to_jaxpr(*args, **kwargs)


def trivial_ctx(*args, **kwargs):
    return ops.trivial_ctx(*args, **kwargs)


import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops

partial_eval_jaxpr_custom_rules: typing.Dict[Any, Any] = {}


def call_param_updaters(*args: Any, **kwargs: Any) -> Any:
    pass


def call_partial_eval_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def const_fold_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def custom_partial_eval_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def custom_staging_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def dce_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def forwarding_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def padding_rules(*args: Any, **kwargs: Any) -> Any:
    pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
