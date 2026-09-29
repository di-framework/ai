#include <metal_stdlib>
using namespace metal;

kernel void unary_f32(
    device const float *X [[buffer(0)]],
    device float *Y [[buffer(1)]],
    constant uint &op [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) {
        return;
    }
    float x = X[i];
    float y;
    switch (op) {
        case 0: y = max(x, 0.0f); break;
        case 1: y = 1.0f / (1.0f + exp(-x)); break;
        case 2: y = tanh(x); break;
        case 3: y = -x; break;
        default: y = x * scale; break;
    }
    Y[i] = y;
}

kernel void binop_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant uint &op [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) {
        return;
    }
    float a = A[i];
    float b = B[i];
    float y;
    switch (op) {
        case 0: y = a + b; break;
        case 1: y = a - b; break;
        case 2: y = a * b; break;
        default: y = a / b; break;
    }
    Y[i] = y;
}

kernel void matmul_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

kernel void softmax_rows_f32(
    device const float *X [[buffer(0)]],
    device float *Y [[buffer(1)]],
    constant uint &C [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    device const float *x = X + row * C;
    device float *y = Y + row * C;
    float m = -INFINITY;
    for (uint i = 0; i < C; ++i) {
        m = max(m, x[i]);
    }
    float s = 0.0f;
    for (uint i = 0; i < C; ++i) {
        y[i] = exp(x[i] - m);
        s += y[i];
    }
    float inv = 1.0f / s;
    for (uint i = 0; i < C; ++i) {
        y[i] *= inv;
    }
}

kernel void matmul_batched_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &BATCH [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    uint b = gid.z;
    if (b >= BATCH || row >= M || col >= N) {
        return;
    }
    device const float *a = A + b * M * K;
    device const float *bb = B + b * K * N;
    device float *c = C + b * M * N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += a[row * K + k] * bb[k * N + col];
    }
    c[row * N + col] = acc;
}

// One thread per (batch, head, query row). Fused online softmax — does not
// materialize the Tq×Tk score matrix.
kernel void sdpa_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *M [[buffer(3)]],
    device float *O [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &H [[buffer(6)]],
    constant uint &Tq [[buffer(7)]],
    constant uint &Tk [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant uint &has_mask [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint q = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (b >= B || h >= H || q >= Tq) {
        return;
    }
    uint q_off = ((b * H + h) * Tq + q) * D;
    float m = -INFINITY;
    for (uint t = 0; t < Tk; ++t) {
        uint k_off = ((b * H + h) * Tk + t) * D;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) {
            s += Q[q_off + d] * K[k_off + d];
        }
        s *= scale;
        if (has_mask) {
            s += M[(((b * H + h) * Tq + q) * Tk) + t];
        }
        m = max(m, s);
    }
    float l = 0.0f;
    for (uint t = 0; t < Tk; ++t) {
        uint k_off = ((b * H + h) * Tk + t) * D;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) {
            s += Q[q_off + d] * K[k_off + d];
        }
        s *= scale;
        if (has_mask) {
            s += M[(((b * H + h) * Tq + q) * Tk) + t];
        }
        l += exp(s - m);
    }
    float inv = 1.0f / l;
    for (uint d = 0; d < D; ++d) {
        O[q_off + d] = 0.0f;
    }
    for (uint t = 0; t < Tk; ++t) {
        uint k_off = ((b * H + h) * Tk + t) * D;
        uint v_off = k_off;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) {
            s += Q[q_off + d] * K[k_off + d];
        }
        s *= scale;
        if (has_mask) {
            s += M[(((b * H + h) * Tq + q) * Tk) + t];
        }
        float p = exp(s - m) * inv;
        for (uint d = 0; d < D; ++d) {
            O[q_off + d] += p * V[v_off + d];
        }
    }
}
