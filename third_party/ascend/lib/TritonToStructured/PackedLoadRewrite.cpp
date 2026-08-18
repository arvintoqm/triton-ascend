#include "TritonToStructured/PackedLoadRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {
struct StaticTensor {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> values;
};

static FailureOr<StaticTensor> evaluate(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrIndex())
    return failure();
  StaticTensor result;
  result.shape.assign(type.getShape().begin(), type.getShape().end());
  result.values.resize(type.getNumElements());

  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    if (result.shape.size() != 1 || range.getStart() < 0 ||
        range.getEnd() - range.getStart() != result.shape[0])
      return failure();
    for (int64_t i = 0; i < result.shape[0]; ++i)
      result.values[i] = range.getStart() + i;
    return result;
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
    if (!dense || dense.getNumElements() != result.values.size())
      return failure();
    for (auto it : llvm::enumerate(dense.getValues<APInt>()))
      result.values[it.index()] = it.value().getSExtValue();
    return result;
  }
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    auto scalar = splat.getSrc();
    auto scalarConst = scalar.getDefiningOp<arith::ConstantIntOp>();
    if (!scalarConst)
      return failure();
    std::fill(result.values.begin(), result.values.end(),
              scalarConst.value());
    return result;
  }
  if (auto expand = value.getDefiningOp<ExpandDimsOp>()) {
    auto source = evaluate(expand.getSrc());
    if (failed(source))
      return failure();
    auto axis = expand.getAxis();
    if (axis < 0 || axis > static_cast<int64_t>(source->shape.size()))
      return failure();
    SmallVector<int64_t> expected;
    expected.append(source->shape.begin(), source->shape.begin() + axis);
    expected.push_back(1);
    expected.append(source->shape.begin() + axis, source->shape.end());
    if (expected != result.shape)
      return failure();
    result.values = source->values;
    return result;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto source = evaluate(broadcast.getSrc());
    if (failed(source) || source->shape.size() > result.shape.size())
      return failure();
    size_t leading = result.shape.size() - source->shape.size();
    for (size_t i = 0; i < source->shape.size(); ++i) {
      if (source->shape[i] != 1 && source->shape[i] != result.shape[leading + i])
        return failure();
    }
    SmallVector<int64_t> strides(result.shape.size(), 1);
    for (int64_t i = result.shape.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * result.shape[i + 1];
    SmallVector<int64_t> sourceStrides(source->shape.size(), 1);
    for (int64_t i = source->shape.size() - 2; i >= 0; --i)
      sourceStrides[i] = sourceStrides[i + 1] * source->shape[i + 1];
    for (int64_t linear = 0; linear < static_cast<int64_t>(result.values.size()); ++linear) {
      int64_t sourceLinear = 0;
      int64_t remainder = linear;
      for (size_t i = 0; i < result.shape.size(); ++i) {
        int64_t coord = remainder / strides[i];
        remainder %= strides[i];
        if (i >= leading && source->shape[i - leading] != 1)
          sourceLinear += coord * sourceStrides[i - leading];
      }
      result.values[linear] = source->values[sourceLinear];
    }
    return result;
  }
  auto binary = [&](Value lhs, Value rhs, auto operation) -> FailureOr<StaticTensor> {
    auto left = evaluate(lhs);
    auto right = evaluate(rhs);
    if (failed(left) || failed(right) || left->shape != right->shape)
      return failure();
    StaticTensor combined = *left;
    for (size_t i = 0; i < combined.values.size(); ++i)
      combined.values[i] = operation(left->values[i], right->values[i]);
    return combined;
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return binary(add.getLhs(), add.getRhs(), [](int64_t a, int64_t b) { return a + b; });
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return binary(mul.getLhs(), mul.getRhs(), [](int64_t a, int64_t b) { return a * b; });
  if (auto band = value.getDefiningOp<arith::AndIOp>())
    return binary(band.getLhs(), band.getRhs(), [](int64_t a, int64_t b) { return a & b; });
  if (auto shr = value.getDefiningOp<arith::ShRSIOp>())
    return binary(shr.getLhs(), shr.getRhs(), [](int64_t a, int64_t b) { return a >> b; });
  return failure();
}

static Value scalarBase(Value value) {
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getSrc();
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return scalarBase(broadcast.getSrc());
  return value;
}

static bool isW3QH(ArrayRef<int64_t> offsets, int64_t rows, int64_t columns) {
  if (columns % 32 != 0)
    return false;
  int64_t groups = columns / 32;
  if (static_cast<int64_t>(offsets.size()) != rows * columns)
    return false;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = (row * groups + (col >> 5)) * 4 + ((col & 31) >> 3);
      if (offsets[row * columns + col] != expected)
        return false;
    }
  return true;
}

static FailureOr<int64_t> isW3QS(ArrayRef<int64_t> offsets, int64_t rows,
                                 int64_t columns) {
  if (columns % 32 != 0 || offsets.empty())
    return failure();
  int64_t groups = columns / 32;
  int64_t pair = offsets[0] & 1;
  if (pair > 1 || static_cast<int64_t>(offsets.size()) != rows * columns)
    return failure();
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = (row * groups + (col >> 5)) * 8 +
                         ((col & 31) >> 3) * 2 + pair;
      if (offsets[row * columns + col] != expected)
        return failure();
    }
  return pair;
}

static bool isW4(ArrayRef<int64_t> offsets, int64_t rows, int64_t columns) {
  if (columns % 32 != 0 || static_cast<int64_t>(offsets.size()) != rows * columns)
    return false;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t col = 0; col < columns; ++col) {
      int64_t expected = row * (columns / 2) + (col >> 5) * 16 + (col & 15);
      if (offsets[row * columns + col] != expected)
        return false;
    }
  return true;
}

static Value createCompactLoad(Location loc, Value base, int64_t elements,
                               PatternRewriter &rewriter) {
  auto basePtr = dyn_cast<PointerType>(base.getType());
  if (!basePtr)
    return nullptr;
  auto ptrType = RankedTensorType::get({elements}, basePtr);
  auto indexType = RankedTensorType::get({elements}, rewriter.getI32Type());
  auto range = rewriter.create<MakeRangeOp>(loc, indexType, 0, elements);
  auto splat = rewriter.create<SplatOp>(loc, ptrType, base);
  auto ptr = rewriter.create<AddPtrOp>(loc, ptrType, splat, range);
  return rewriter.create<LoadOp>(loc, ptr, nullptr, nullptr, nullptr, nullptr,
                                 false).getResult();
}
} // namespace

LogicalResult PackedLoadRewrite::matchAndRewrite(
    LoadOp op, PatternRewriter &rewriter) const {
  if (op.getMask() || op.getOther())
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto addptr = op.getPtr().getDefiningOp<AddPtrOp>();
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() != 2 ||
      !addptr)
    return failure();
  auto offsets = evaluate(addptr.getOffset());
  if (failed(offsets) || offsets->shape != resultType.getShape())
    return failure();
  int64_t rows = resultType.getShape()[0];
  int64_t columns = resultType.getShape()[1];
  bool w3qh = isW3QH(offsets->values, rows, columns);
  auto w3qsPair = isW3QS(offsets->values, rows, columns);
  bool w4 = isW4(offsets->values, rows, columns);
  if (!w3qh && failed(w3qsPair) && !w4)
    return failure();
  bool w3qs = succeeded(w3qsPair);
  int64_t physical = (w3qh || w3qs) ? rows * (columns / 32) * (w3qs ? 8 : 4)
                          : rows * (columns / 2);
  Value compact = createCompactLoad(op.getLoc(), scalarBase(addptr.getPtr()),
                                     physical, rewriter);
  if (!compact)
    return failure();
  auto compactType = cast<RankedTensorType>(compact.getType());
    SmallVector<int64_t> packedShape =
      (w3qh || w3qs) ? SmallVector<int64_t>{rows, columns / 32, w3qs ? 4 : 4}
           : SmallVector<int64_t>{rows, columns / 32, 16};
    Value packedInput = compact;
    if (w3qs) {
    packedShape.push_back(2);
    auto packedFull = rewriter.create<ReshapeOp>(
      op.getLoc(), RankedTensorType::get(packedShape, compactType.getElementType()),
      compact);
    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    offsets[3] = rewriter.getIndexAttr(*w3qsPair);
    SmallVector<OpFoldResult> sizes = {
      rewriter.getIndexAttr(rows), rewriter.getIndexAttr(columns / 32),
      rewriter.getIndexAttr(4), rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    auto sliceType = RankedTensorType::get(
      {rows, columns / 32, 4, 1}, compactType.getElementType());
    packedInput = rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), sliceType, packedFull.getResult(), offsets, sizes,
      strides);
    packedShape.pop_back();
    }
  auto packed = rewriter.create<ReshapeOp>(op.getLoc(),
                                           RankedTensorType::get(packedShape, compactType.getElementType()),
                         packedInput);
    int64_t broadcastAxis = (w3qh || w3qs) ? 3 : 2;
  auto expanded = rewriter.create<ExpandDimsOp>(
      op.getLoc(), packed.getResult(), broadcastAxis);
  SmallVector<int64_t> expandedShape = packedShape;
  expandedShape.insert(expandedShape.begin() + broadcastAxis, w3qh ? 1 : 1);
  expandedShape[broadcastAxis] = (w3qh || w3qs) ? 8 : 2;
  auto broadcast = rewriter.create<BroadcastOp>(
      op.getLoc(), RankedTensorType::get(expandedShape, compactType.getElementType()),
      expanded.getResult());
  auto restored = rewriter.create<ReshapeOp>(op.getLoc(), resultType,
                                             broadcast.getResult());
  rewriter.replaceOp(op, restored.getResult());
  op.emitRemark() << "PackedLoadRewrite: logical shape=" << rows << "x" << columns
                  << " physical elements=" << physical
                  << " reuse factor=" << (rows * columns) / physical
                  << " kind=" << (w3qh ? "W3 QH" : (w3qs ? "W3 QS" : "W4 qweight"));
  return success();
}
