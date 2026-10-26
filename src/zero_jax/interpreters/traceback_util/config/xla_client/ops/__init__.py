"""Canonical XLA client operation node classes."""

from __future__ import annotations

from zero_jax._types import ConfigValue

Tuple = tuple


class _XlaOpNode:
    """Base class for XLA builder operation descriptors."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ``_XlaOpNode``.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Abs(_XlaOpNode):
    """XLA operation descriptor for ``Abs``."""


class Acos(_XlaOpNode):
    """XLA operation descriptor for ``Acos``."""


class Acosh(_XlaOpNode):
    """XLA operation descriptor for ``Acosh``."""


class Add(_XlaOpNode):
    """XLA operation descriptor for ``Add``."""


class AfterAll(_XlaOpNode):
    """XLA operation descriptor for ``AfterAll``."""


class AllGather(_XlaOpNode):
    """XLA operation descriptor for ``AllGather``."""


class AllReduce(_XlaOpNode):
    """XLA operation descriptor for ``AllReduce``."""


class AllToAll(_XlaOpNode):
    """XLA operation descriptor for ``AllToAll``."""


class And(_XlaOpNode):
    """XLA operation descriptor for ``And``."""


class ApproxTopK(_XlaOpNode):
    """XLA operation descriptor for ``ApproxTopK``."""


class ApproxTopKFallback(_XlaOpNode):
    """XLA operation descriptor for ``ApproxTopKFallback``."""


class ApproxTopKReductionOutputSize(_XlaOpNode):
    """XLA operation descriptor for ``ApproxTopKReductionOutputSize``."""


class Asin(_XlaOpNode):
    """XLA operation descriptor for ``Asin``."""


class Asinh(_XlaOpNode):
    """XLA operation descriptor for ``Asinh``."""


class Atan(_XlaOpNode):
    """XLA operation descriptor for ``Atan``."""


class Atan2(_XlaOpNode):
    """XLA operation descriptor for ``Atan2``."""


class Atanh(_XlaOpNode):
    """XLA operation descriptor for ``Atanh``."""


class BesselI0e(_XlaOpNode):
    """XLA operation descriptor for ``BesselI0e``."""


class BesselI1e(_XlaOpNode):
    """XLA operation descriptor for ``BesselI1e``."""


class BitcastConvertType(_XlaOpNode):
    """XLA operation descriptor for ``BitcastConvertType``."""


class Broadcast(_XlaOpNode):
    """XLA operation descriptor for ``Broadcast``."""


class BroadcastInDim(_XlaOpNode):
    """XLA operation descriptor for ``BroadcastInDim``."""


class Call(_XlaOpNode):
    """XLA operation descriptor for ``Call``."""


class Cbrt(_XlaOpNode):
    """XLA operation descriptor for ``Cbrt``."""


class Ceil(_XlaOpNode):
    """XLA operation descriptor for ``Ceil``."""


class Cholesky(_XlaOpNode):
    """XLA operation descriptor for ``Cholesky``."""


class Clamp(_XlaOpNode):
    """XLA operation descriptor for ``Clamp``."""


class Clz(_XlaOpNode):
    """XLA operation descriptor for ``Clz``."""


class Collapse(_XlaOpNode):
    """XLA operation descriptor for ``Collapse``."""


class CollectivePermute(_XlaOpNode):
    """XLA operation descriptor for ``CollectivePermute``."""


class Complex(_XlaOpNode):
    """XLA operation descriptor for ``Complex``."""


class ConcatInDim(_XlaOpNode):
    """XLA operation descriptor for ``ConcatInDim``."""


class Conditional(_XlaOpNode):
    """XLA operation descriptor for ``Conditional``."""


class Conj(_XlaOpNode):
    """XLA operation descriptor for ``Conj``."""


class Constant(_XlaOpNode):
    """XLA operation descriptor for ``Constant``."""


class ConstantLiteral(_XlaOpNode):
    """XLA operation descriptor for ``ConstantLiteral``."""


class ConvGeneralDilated(_XlaOpNode):
    """XLA operation descriptor for ``ConvGeneralDilated``."""


class ConvertElementType(_XlaOpNode):
    """XLA operation descriptor for ``ConvertElementType``."""


class Cos(_XlaOpNode):
    """XLA operation descriptor for ``Cos``."""


class Cosh(_XlaOpNode):
    """XLA operation descriptor for ``Cosh``."""


class CreateToken(_XlaOpNode):
    """XLA operation descriptor for ``CreateToken``."""


class CrossReplicaSum(_XlaOpNode):
    """XLA operation descriptor for ``CrossReplicaSum``."""


class CustomCall(_XlaOpNode):
    """XLA operation descriptor for ``CustomCall``."""


class CustomCallApiVersion(_XlaOpNode):
    """XLA operation descriptor for ``CustomCallApiVersion``."""


class CustomCallSchedule(_XlaOpNode):
    """XLA operation descriptor for ``CustomCallSchedule``."""


class CustomCallWithAliasing(_XlaOpNode):
    """XLA operation descriptor for ``CustomCallWithAliasing``."""


class CustomCallWithComputation(_XlaOpNode):
    """XLA operation descriptor for ``CustomCallWithComputation``."""


class CustomCallWithLayout(_XlaOpNode):
    """XLA operation descriptor for ``CustomCallWithLayout``."""


class Digamma(_XlaOpNode):
    """XLA operation descriptor for ``Digamma``."""


class Div(_XlaOpNode):
    """XLA operation descriptor for ``Div``."""


class Dot(_XlaOpNode):
    """XLA operation descriptor for ``Dot``."""


class DotGeneral(_XlaOpNode):
    """XLA operation descriptor for ``DotGeneral``."""


class DynamicReshape(_XlaOpNode):
    """XLA operation descriptor for ``DynamicReshape``."""


class DynamicSlice(_XlaOpNode):
    """XLA operation descriptor for ``DynamicSlice``."""


class DynamicUpdateSlice(_XlaOpNode):
    """XLA operation descriptor for ``DynamicUpdateSlice``."""


class Eigh(_XlaOpNode):
    """XLA operation descriptor for ``Eigh``."""


class Eq(_XlaOpNode):
    """XLA operation descriptor for ``Eq``."""


class Erf(_XlaOpNode):
    """XLA operation descriptor for ``Erf``."""


class ErfInv(_XlaOpNode):
    """XLA operation descriptor for ``ErfInv``."""


class Erfc(_XlaOpNode):
    """XLA operation descriptor for ``Erfc``."""


class Exp(_XlaOpNode):
    """XLA operation descriptor for ``Exp``."""


class Expm1(_XlaOpNode):
    """XLA operation descriptor for ``Expm1``."""


class Fft(_XlaOpNode):
    """XLA operation descriptor for ``Fft``."""


class Floor(_XlaOpNode):
    """XLA operation descriptor for ``Floor``."""


class Gather(_XlaOpNode):
    """XLA operation descriptor for ``Gather``."""


class Ge(_XlaOpNode):
    """XLA operation descriptor for ``Ge``."""


class GetDimensionSize(_XlaOpNode):
    """XLA operation descriptor for ``GetDimensionSize``."""


class GetTupleElement(_XlaOpNode):
    """XLA operation descriptor for ``GetTupleElement``."""


class Gt(_XlaOpNode):
    """XLA operation descriptor for ``Gt``."""


class Igamma(_XlaOpNode):
    """XLA operation descriptor for ``Igamma``."""


class IgammaGradA(_XlaOpNode):
    """XLA operation descriptor for ``IgammaGradA``."""


class Igammac(_XlaOpNode):
    """XLA operation descriptor for ``Igammac``."""


class Imag(_XlaOpNode):
    """XLA operation descriptor for ``Imag``."""


class InfeedWithToken(_XlaOpNode):
    """XLA operation descriptor for ``InfeedWithToken``."""


class Iota(_XlaOpNode):
    """XLA operation descriptor for ``Iota``."""


class IsFinite(_XlaOpNode):
    """XLA operation descriptor for ``IsFinite``."""


class LU(_XlaOpNode):
    """XLA operation descriptor for ``LU``."""


class Le(_XlaOpNode):
    """XLA operation descriptor for ``Le``."""


class Lgamma(_XlaOpNode):
    """XLA operation descriptor for ``Lgamma``."""


class Log(_XlaOpNode):
    """XLA operation descriptor for ``Log``."""


class Log1p(_XlaOpNode):
    """XLA operation descriptor for ``Log1p``."""


class Lt(_XlaOpNode):
    """XLA operation descriptor for ``Lt``."""


class Map(_XlaOpNode):
    """XLA operation descriptor for ``Map``."""


class Max(_XlaOpNode):
    """XLA operation descriptor for ``Max``."""


class Min(_XlaOpNode):
    """XLA operation descriptor for ``Min``."""


class Mul(_XlaOpNode):
    """XLA operation descriptor for ``Mul``."""


class Ne(_XlaOpNode):
    """XLA operation descriptor for ``Ne``."""


class Neg(_XlaOpNode):
    """XLA operation descriptor for ``Neg``."""


class NextAfter(_XlaOpNode):
    """XLA operation descriptor for ``NextAfter``."""


class Not(_XlaOpNode):
    """XLA operation descriptor for ``Not``."""


class OptimizationBarrier(_XlaOpNode):
    """XLA operation descriptor for ``OptimizationBarrier``."""


class Or(_XlaOpNode):
    """XLA operation descriptor for ``Or``."""


class OutfeedWithToken(_XlaOpNode):
    """XLA operation descriptor for ``OutfeedWithToken``."""


class Pad(_XlaOpNode):
    """XLA operation descriptor for ``Pad``."""


class Parameter(_XlaOpNode):
    """XLA operation descriptor for ``Parameter``."""


class PopulationCount(_XlaOpNode):
    """XLA operation descriptor for ``PopulationCount``."""


class Pow(_XlaOpNode):
    """XLA operation descriptor for ``Pow``."""


class ProductOfElementaryHouseholderReflectors(_XlaOpNode):
    """XLA operation descriptor for ``ProductOfElementaryHouseholderReflectors``."""


class QR(_XlaOpNode):
    """XLA operation descriptor for ``QR``."""


class QrDecomposition(_XlaOpNode):
    """XLA operation descriptor for ``QrDecomposition``."""


class RandomAlgorithm(_XlaOpNode):
    """XLA operation descriptor for ``RandomAlgorithm``."""


class RandomGammaGrad(_XlaOpNode):
    """XLA operation descriptor for ``RandomGammaGrad``."""


class Real(_XlaOpNode):
    """XLA operation descriptor for ``Real``."""


class Reciprocal(_XlaOpNode):
    """XLA operation descriptor for ``Reciprocal``."""


class RecvFromHost(_XlaOpNode):
    """XLA operation descriptor for ``RecvFromHost``."""


class Reduce(_XlaOpNode):
    """XLA operation descriptor for ``Reduce``."""


class ReducePrecision(_XlaOpNode):
    """XLA operation descriptor for ``ReducePrecision``."""


class ReduceScatter(_XlaOpNode):
    """XLA operation descriptor for ``ReduceScatter``."""


class ReduceWindowWithGeneralPadding(_XlaOpNode):
    """XLA operation descriptor for ``ReduceWindowWithGeneralPadding``."""


class RegularizedIncompleteBeta(_XlaOpNode):
    """XLA operation descriptor for ``RegularizedIncompleteBeta``."""


class Rem(_XlaOpNode):
    """XLA operation descriptor for ``Rem``."""


class RemoveDynamicDimension(_XlaOpNode):
    """XLA operation descriptor for ``RemoveDynamicDimension``."""


class ReplicaId(_XlaOpNode):
    """XLA operation descriptor for ``ReplicaId``."""


class Reshape(_XlaOpNode):
    """XLA operation descriptor for ``Reshape``."""


class Rev(_XlaOpNode):
    """XLA operation descriptor for ``Rev``."""


class RngBitGenerator(_XlaOpNode):
    """XLA operation descriptor for ``RngBitGenerator``."""


class RngNormal(_XlaOpNode):
    """XLA operation descriptor for ``RngNormal``."""


class RngUniform(_XlaOpNode):
    """XLA operation descriptor for ``RngUniform``."""


class Round(_XlaOpNode):
    """XLA operation descriptor for ``Round``."""


class Rsqrt(_XlaOpNode):
    """XLA operation descriptor for ``Rsqrt``."""


class SVD(_XlaOpNode):
    """XLA operation descriptor for ``SVD``."""


class Scatter(_XlaOpNode):
    """XLA operation descriptor for ``Scatter``."""


class Select(_XlaOpNode):
    """XLA operation descriptor for ``Select``."""


class SelectAndScatterWithGeneralPadding(_XlaOpNode):
    """XLA operation descriptor for ``SelectAndScatterWithGeneralPadding``."""


class SendToHost(_XlaOpNode):
    """XLA operation descriptor for ``SendToHost``."""


class SetDimensionSize(_XlaOpNode):
    """XLA operation descriptor for ``SetDimensionSize``."""


class ShiftLeft(_XlaOpNode):
    """XLA operation descriptor for ``ShiftLeft``."""


class ShiftRightArithmetic(_XlaOpNode):
    """XLA operation descriptor for ``ShiftRightArithmetic``."""


class ShiftRightLogical(_XlaOpNode):
    """XLA operation descriptor for ``ShiftRightLogical``."""


class Sign(_XlaOpNode):
    """XLA operation descriptor for ``Sign``."""


class Sin(_XlaOpNode):
    """XLA operation descriptor for ``Sin``."""


class Sinh(_XlaOpNode):
    """XLA operation descriptor for ``Sinh``."""


class Slice(_XlaOpNode):
    """XLA operation descriptor for ``Slice``."""


class SliceInDim(_XlaOpNode):
    """XLA operation descriptor for ``SliceInDim``."""


class Sort(_XlaOpNode):
    """XLA operation descriptor for ``Sort``."""


class Sqrt(_XlaOpNode):
    """XLA operation descriptor for ``Sqrt``."""


class Square(_XlaOpNode):
    """XLA operation descriptor for ``Square``."""


class Sub(_XlaOpNode):
    """XLA operation descriptor for ``Sub``."""


class Tan(_XlaOpNode):
    """XLA operation descriptor for ``Tan``."""


class Tanh(_XlaOpNode):
    """XLA operation descriptor for ``Tanh``."""


class TopK(_XlaOpNode):
    """XLA operation descriptor for ``TopK``."""


class Transpose(_XlaOpNode):
    """XLA operation descriptor for ``Transpose``."""


class TriangularSolve(_XlaOpNode):
    """XLA operation descriptor for ``TriangularSolve``."""


class TriangularSolveOptions_Transpose(_XlaOpNode):
    """XLA operation descriptor for ``TriangularSolveOptions_Transpose``."""


class While(_XlaOpNode):
    """XLA operation descriptor for ``While``."""


class Xor(_XlaOpNode):
    """XLA operation descriptor for ``Xor``."""


class Zeta(_XlaOpNode):
    """XLA operation descriptor for ``Zeta``."""


__all__ = [
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "AfterAll",
    "AllGather",
    "AllReduce",
    "AllToAll",
    "And",
    "ApproxTopK",
    "ApproxTopKFallback",
    "ApproxTopKReductionOutputSize",
    "Asin",
    "Asinh",
    "Atan",
    "Atan2",
    "Atanh",
    "BesselI0e",
    "BesselI1e",
    "BitcastConvertType",
    "Broadcast",
    "BroadcastInDim",
    "Call",
    "Cbrt",
    "Ceil",
    "Cholesky",
    "Clamp",
    "Clz",
    "Collapse",
    "CollectivePermute",
    "Complex",
    "ConcatInDim",
    "Conditional",
    "Conj",
    "Constant",
    "ConstantLiteral",
    "ConvGeneralDilated",
    "ConvertElementType",
    "Cos",
    "Cosh",
    "CreateToken",
    "CrossReplicaSum",
    "CustomCall",
    "CustomCallApiVersion",
    "CustomCallSchedule",
    "CustomCallWithAliasing",
    "CustomCallWithComputation",
    "CustomCallWithLayout",
    "Digamma",
    "Div",
    "Dot",
    "DotGeneral",
    "DynamicReshape",
    "DynamicSlice",
    "DynamicUpdateSlice",
    "Eigh",
    "Eq",
    "Erf",
    "ErfInv",
    "Erfc",
    "Exp",
    "Expm1",
    "Fft",
    "Floor",
    "Gather",
    "Ge",
    "GetDimensionSize",
    "GetTupleElement",
    "Gt",
    "Igamma",
    "IgammaGradA",
    "Igammac",
    "Imag",
    "InfeedWithToken",
    "Iota",
    "IsFinite",
    "LU",
    "Le",
    "Lgamma",
    "Log",
    "Log1p",
    "Lt",
    "Map",
    "Max",
    "Min",
    "Mul",
    "Ne",
    "Neg",
    "NextAfter",
    "Not",
    "OptimizationBarrier",
    "Or",
    "OutfeedWithToken",
    "Pad",
    "Parameter",
    "PopulationCount",
    "Pow",
    "ProductOfElementaryHouseholderReflectors",
    "QR",
    "QrDecomposition",
    "RandomAlgorithm",
    "RandomGammaGrad",
    "Real",
    "Reciprocal",
    "RecvFromHost",
    "Reduce",
    "ReducePrecision",
    "ReduceScatter",
    "ReduceWindowWithGeneralPadding",
    "RegularizedIncompleteBeta",
    "Rem",
    "RemoveDynamicDimension",
    "ReplicaId",
    "Reshape",
    "Rev",
    "RngBitGenerator",
    "RngNormal",
    "RngUniform",
    "Round",
    "Rsqrt",
    "SVD",
    "Scatter",
    "Select",
    "SelectAndScatterWithGeneralPadding",
    "SendToHost",
    "SetDimensionSize",
    "ShiftLeft",
    "ShiftRightArithmetic",
    "ShiftRightLogical",
    "Sign",
    "Sin",
    "Sinh",
    "Slice",
    "SliceInDim",
    "Sort",
    "Sqrt",
    "Square",
    "Sub",
    "Tan",
    "Tanh",
    "TopK",
    "Transpose",
    "TriangularSolve",
    "TriangularSolveOptions_Transpose",
    "Tuple",
    "While",
    "Xor",
    "Zeta",
    "_XlaOpNode",
]
