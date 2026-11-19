#!/usr/bin/env python3
"""Generates Sources/AetherEngine/Video/Metal/ArtCNNShaderSource.swift from ArtCNN's mpv compute shaders
GLSL/ArtCNN_C4F16.glsl and GLSL/ArtCNN_C4F32.glsl (https://github.com/Artoriuz/ArtCNN, MIT).

    python3 -I Scripts/generate-artcnn-shader.py <ArtCNN checkout>/GLSL

The generated file is checked in; rerun this only to pick up a new revision of the shaders. Each mpv
`//!COMPUTE` hook pass up to Conv2D-6 becomes one Metal compute kernel with the same threadgroup tiling, the
same arithmetic in half precision and the weights unchanged. The feature maps keep the GLSL's packing (16
channels as 2x2 rgba16Float texels per source pixel for C4F16, 32 as 4x2 for C4F32). What differs:

- The first pass reads the decoder's luma plane directly (r8Unorm or r16Unorm), inside the picture region,
  scaled to the code range the way mpv's LUMA_mul does.
- Reads outside the picture repeat its edge sample (mpv's out-of-range texelFetch is undefined).
- The Depth-To-Space pass is replaced by `artcnnOutput`, which writes the 2x2 output luma samples of each
  source pixel straight into the output frame's luma plane and the one chroma sample of that block into
  its chroma plane (bilinear from the source chroma, sampled where `FrameChain` samples it). There is no
  RGB anywhere: ArtCNN is a luma network, as in mpv.

It also writes `artcnnHiddenMatrix`, which runs Conv2D-1...5 on the matrix units (simdgroup_matrix, Apple7 GPUs)
from the same weights, read out of the GLSL's mat4 literals into a float16 table (`<model>HiddenWeights`).
"""
import base64
import os
import re
import struct
import sys

OUT = os.path.join(os.path.dirname(__file__), '..', 'Sources', 'AetherEngine', 'Video', 'Metal',
                   'ArtCNNShaderSource.swift')
REVISION = '0837e5326e710f49e2ab0a8d995da0708047442e'
MODELS = [('c4f16', 'ArtCNN_C4F16.glsl'), ('c4f32', 'ArtCNN_C4F32.glsl')]


def parse(path):
    header, passes, cur = [], [], None
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('//!DESC'):
            cur = {'desc': line[7:].strip(), 'binds': [], 'save': None, 'compute': None,
                   'width': None, 'height': None, 'lines': []}
            passes.append(cur)
        elif cur is None:
            if line.startswith('//'):
                header.append(line[2:].strip())
            elif not line.strip() and header and header[-1]:
                header.append('')
        elif line.startswith('//!BIND'):
            cur['binds'].append(line.split()[1])
        elif line.startswith('//!SAVE'):
            cur['save'] = line.split()[1]
        elif line.startswith('//!COMPUTE'):
            cur['compute'] = [int(v) for v in line.split()[1:5]]
        elif line.startswith('//!WIDTH'):
            cur['width'] = float(line.split()[2])
        elif line.startswith('//!HEIGHT'):
            cur['height'] = float(line.split()[2])
        elif not line.startswith('//!'):
            cur['lines'].append(line)
    while header and not header[-1]:
        header.pop()
    return header, passes


def hook_body(p):
    """The statements of `void hook() { ... }` and the threadgroup array declarations before it."""
    text = '\n'.join(p['lines'])
    shared = re.findall(r'^shared (\w+) (inp\[[^;]*\]);', text, flags=re.M)
    m = re.search(r'void hook\(\)\s*\{\n(.*)\n\}\s*$', text, flags=re.S)
    assert m, p['desc']
    return shared, m.group(1)


def translate(p, index, packing):
    """One conv pass as a Metal kernel. `packing` is (px, py): texels per source pixel of the features."""
    bw, bh, tw, th = p['compute']
    shared, body = hook_body(p)
    isize = (tw + 2, th + 2)
    binds = p['binds']
    first = binds == ['LUMA']
    # The pass's own output packing: per thread, output_base = gid * (ox, oy).
    om = re.search(r'output_base = ivec2\(gl_GlobalInvocationID\) \* ivec2\((\d+), (\d+)\)', body)
    assert om, p['desc']
    if not first:
        im = re.search(r'input_base = \(base \+ ivec2\(x,y\) - offset\) \* ivec2\((\d+), (\d+)\)', body)
        assert im and (int(im.group(1)), int(im.group(2))) == packing, p['desc']

    # The input coordinate: clamped to the source picture in source pixels, then packed.
    body = re.sub(r'const ivec2 input_base = \(base \+ ivec2\(x,y\) - offset\) \* ivec2\((\d+), (\d+)\);',
                  lambda mm: ('const int2 input_base = origin + clamp(base + int2(x, y) - offset, int2(0), lrMax);'
                              if first else
                              f'const int2 input_base = clamp(base + int2(x, y) - offset, int2(0), lrMax) * int2({mm.group(1)}, {mm.group(2)});'),
                  body)
    body = body.replace('LUMA_mul * texelFetch(LUMA_raw, input_base + ivec2(0, 0), 0).x',
                        'P.lumaMul * LUMA.read(uint2(input_base)).r')
    for b in binds:
        if b == 'LUMA':
            continue
        body = re.sub(b + r'_mul \* texelFetch\(' + b + r'_raw, (input_base \+ ivec2\(\d, \d\)), 0\)',
                      lambda mm, b=b: f'{b}.read(uint2({mm.group(1)}))', body)
    assert 'texelFetch' not in body and '_mul' not in body, p['desc']

    body = body.replace('const uvec2 local_xy = gl_LocalInvocationID.xy;', 'const uint2 local_xy = lid;')
    body = body.replace('ivec2 base = ivec2(gl_WorkGroupID) * wg_size;', 'const int2 base = int2(tgid) * wg_size;')
    body = body.replace('const ivec2 output_base = ivec2(gl_GlobalInvocationID)', 'const int2 output_base = int2(gid)')
    body = body.replace('for (uint y = local_xy.y; y < isize.y; y += wg_size.y)',
                        f'for (uint y = local_xy.y; y < {isize[1]}u; y += {th}u)')
    body = body.replace('for (uint x = local_xy.x; x < isize.x; x += wg_size.x)',
                        f'for (uint x = local_xy.x; x < {isize[0]}u; x += {tw}u)')
    body = body.replace('barrier();', 'threadgroup_barrier(mem_flags::mem_threadgroup);')
    # A thread past the picture still loads its share of the tile, but writes nothing.
    body = re.sub(r'imageStore\(out_image, (output_base \+ ivec2\(\d, \d\)), (.*)\);',
                  lambda mm: f'if (inside) out_image.write({mm.group(2)}, uint2({mm.group(1)}));', body)
    body = body.replace('ivec2(', 'int2(').replace('V4(0.0)', 'V4(0.0h)')
    assert 'imageStore' not in body and 'gl_' not in body and 'ivec2' not in body, p['desc']

    decls = '\n'.join(f'    threadgroup {t} {arr.replace("isize.y", str(isize[1])).replace("isize.x", str(isize[0]))};'
                      for t, arr in shared)
    args = []
    if first:
        args.append('texture2d<float, access::read> LUMA [[texture(0)]]')
    else:
        for i, b in enumerate(binds):
            args.append(f'texture2d<half, access::read> {b} [[texture({i})]]')
    args.append(f'texture2d<half, access::write> out_image [[texture({len(args)})]]')
    args += ['constant ArtCNNParams &P [[buffer(1)]]',
             'uint2 gid [[thread_position_in_grid]]',
             'uint2 lid [[thread_position_in_threadgroup]]',
             'uint2 tgid [[threadgroup_position_in_grid]]']
    head = (f"// {p['desc']}: threadgroups of {tw}x{th} threads, one source pixel each.\n"
            f"kernel void artcnnConv{index}({', '.join(args)}) {{\n"
            f"    constexpr int2 wg_size = int2({tw}, {th});\n"
            f"    constexpr int2 offset = int2(1, 1);\n"
            f"    const int2 lrMax = int2(P.width, P.height) - 1;\n"
            f"    const int2 origin = int2(P.originX, P.originY);\n"
            f"    const bool inside = gid.x < P.width && gid.y < P.height;\n"
            f"{decls}\n")
    return head + body + '\n}\n', (tw, th)


PRELUDE = r"""#include <metal_stdlib>
using namespace metal;

#define F half
#define V4 half4
#define M4 artcnn_m4

// GLSL's mat4(16 scalars) is column-major, and so is this.
static inline half4x4 artcnn_m4(float a, float b, float c, float d, float e, float f, float g, float h,
                                float i, float j, float k, float l, float m, float n, float o, float p) {
    return half4x4(half4(a, b, c, d), half4(e, f, g, h), half4(i, j, k, l), half4(m, n, o, p));
}

// Layout matches `ArtCNNParams` in ArtCNNStage.swift.
struct ArtCNNParams {
    uint originX; uint originY;   // the picture region in the input planes (even)
    uint width; uint height;      // its size: the network's source size
    float lumaMul;                // luma plane read -> code / maxCode, mpv's LUMA_mul
    float codeScale;              // plane read * codeScale = code
    float maxCode;                // 255 or 1023
};

inline float artcnn_quantize(float v, constant ArtCNNParams &P) {
    return clamp(rint(v * P.maxCode), 0.0, P.maxCode) / P.codeScale;
}

// Conv2D-6's four channels are the 2x2 output luma samples of each source pixel (mpv's Depth-To-Space:
// channel (dy * 2 + dx)), clamped to 0...1 and written as whole codes. The block's chroma sample is a
// bilinear read of the source chroma at the block's centre: where FrameChain samples it for the same pixel.
kernel void artcnnOutput(texture2d<half, access::read> conv2d_6 [[texture(0)]],
                         texture2d<float, access::sample> chroma [[texture(1)]],
                         texture2d<float, access::write> outLuma [[texture(2)]],
                         texture2d<float, access::write> outChroma [[texture(3)]],
                         constant ArtCNNParams &P [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    constexpr sampler bilinear(coord::pixel, address::clamp_to_edge, filter::linear);
    const float4 r = float4(conv2d_6.read(gid));
    const uint2 o = gid * 2;
    outLuma.write(float4(artcnn_quantize(r.x, P)), o);
    outLuma.write(float4(artcnn_quantize(r.y, P)), o + uint2(1, 0));
    outLuma.write(float4(artcnn_quantize(r.z, P)), o + uint2(0, 1));
    outLuma.write(float4(artcnn_quantize(r.w, P)), o + uint2(1, 1));
    const float2 cpos = (float2(gid + uint2(P.originX, P.originY)) + 0.5) * 0.5;
    const float2 c = chroma.sample(bilinear, cpos).rg * P.codeScale;
    outChroma.write(float4(clamp(rint(c), 0.0, P.maxCode) / P.codeScale, 0.0, 1.0), gid);
}
"""


def layer_weights(p, n_in, n_out, scalar_input=False):
    """A pass's weights as [tap][in][out] and its bias [out], from the GLSL's literals, and whether it ends in
    a ReLU. `resultJ += M4(16 scalars) * inp_K_DX_DY` adds output channels 4J..4J+3 from input channels
    4K..4K+3 at tap (DX, DY); mat4 is column-major, so scalar 4C+R is weight (in 4K+C, out 4J+R). The first
    pass has one input channel: `resultJ += V4(4 scalars) * inp_0_DX_DY`."""
    _, body = hook_body(p)
    w = [[[None] * n_out for _ in range(n_in)] for _ in range(9)]
    if scalar_input:
        for m in re.finditer(r'result(\d+) \+= V4\(([^)]*)\) \* inp_0_(\d)_(\d);', body):
            j, dx, dy = int(m.group(1)), int(m.group(3)), int(m.group(4))
            for r, v in enumerate(float(x) for x in m.group(2).split(',')):
                assert w[dy * 3 + dx][0][4 * j + r] is None
                w[dy * 3 + dx][0][4 * j + r] = v
    else:
        for m in re.finditer(r'result(\d+) \+= M4\(([^)]*)\) \* inp_(\d+)_(\d)_(\d);', body):
            j, k, dx, dy = int(m.group(1)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            vals = [float(v) for v in m.group(2).split(',')]
            assert len(vals) == 16
            for c in range(4):
                for r in range(4):
                    assert w[dy * 3 + dx][4 * k + c][4 * j + r] is None
                    w[dy * 3 + dx][4 * k + c][4 * j + r] = vals[c * 4 + r]
    assert all(v is not None for t in w for row in t for v in row), p['desc']
    bias = [None] * n_out
    for m in re.finditer(r'V4 result(\d+) = V4\(([^)]*)\);', body):
        for r, v in enumerate(float(x) for x in m.group(2).split(',')):
            bias[4 * int(m.group(1)) + r] = v
    assert all(v is not None for v in bias), p['desc']
    return w, bias, 'max(result0, V4(0.0))' in body


def halves(values):
    return struct.pack(f'<{len(values)}e', *values)


def hidden_weights(passes, n):
    """Conv2D-1...5's weights for `artcnnHiddenMatrix`, float16: per pass [tap][in][out], then the bias [out]."""
    blob = b''
    relus = []
    for p in passes[1:6]:
        w, b, relu = layer_weights(p, n, n)
        blob += halves([v for t in w for row in t for v in row]) + halves(b)
        relus.append(relu)
    assert relus == [True, True, True, True, False], relus
    return blob


MATRIX_KERNEL = r"""
// Conv2D-1...5 on the matrix units (Apple7 and later: simdgroup_matrix), the same arithmetic as artcnnConv1...5
// reordered: each 3x3 tap of 8 neighbouring source pixels is an (8 x N) x (N x N) product, accumulated over the
// 9 taps in half precision. A threadgroup of 4 simdgroups takes a 16x8-pixel tile: it loads the tile and a
// 1-pixel margin (edge samples repeated) from the packed feature texture into threadgroup memory as N halves
// per pixel, multiplies there, stores the products, and writes them back packed with the bias and, for the
// ReLU passes, the max with 0. Weights: `hiddenWeights`, per pass [tap][in][out] then the bias.
//
// Measured on an M2 Pro (16-core GPU), 1080p: 2.6 ms a pass at N = 16 against 3.9 for artcnnConv1...5. Reading
// the matrices straight from device memory was 3x slower, and the weights staged in threadgroup memory too
// were slower than reading them from device memory, where they stay cached.
constant bool artcnnRelu [[function_constant(0)]];

kernel void artcnnHiddenMatrix(texture2d<half, access::read> src [[texture(0)]],
                               texture2d<half, access::write> dst [[texture(1)]],
                               constant ArtCNNParams &P [[buffer(1)]],
                               device const half *weights [[buffer(2)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int N = ARTCNN_N, G = N / 4, PX = ARTCNN_PX, TX = 16, TY = 8, THREADS = 128;
    constexpr int SEGS = TY * (TX / 8) / 4;   // 8-pixel row segments per simdgroup
    threadgroup half tile[(TY + 2) * (TX + 2) * N];
    threadgroup half products[TY * TX * N];
    const int2 base = int2(tgid) * int2(TX, TY);
    const int2 lrMax = int2(P.width, P.height) - 1;
    for (int i = tid; i < (TX + 2) * (TY + 2) * G; i += THREADS) {
        const int g = i % G, p = i / G;
        const int2 pix = clamp(base + int2(p % (TX + 2), p / (TX + 2)) - 1, int2(0), lrMax);
        *(threadgroup half4 *)(tile + p * N + g * 4) = src.read(uint2(pix.x * PX + g % PX, pix.y * 2 + g / PX));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if ARTCNN_N <= 16
    // At N = 16 every weight and accumulator of a tap fits in registers (fastest on an M2 Pro).
    simdgroup_half8x8 acc[SEGS][N / 8];
    for (int s = 0; s < SEGS; s++)
        for (int j = 0; j < N / 8; j++) acc[s][j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (int t = 0; t < 9; t++) {
        const int dx = t % 3, dy = t / 3;
        simdgroup_half8x8 w[N / 8][N / 8];
        for (int k = 0; k < N / 8; k++)
            for (int j = 0; j < N / 8; j++) simdgroup_load(w[k][j], weights + t * N * N + k * 8 * N + j * 8, N);
        for (int s = 0; s < SEGS; s++) {
            const int seg = sg * SEGS + s, row = seg / (TX / 8), col = (seg % (TX / 8)) * 8;
            for (int k = 0; k < N / 8; k++) {
                simdgroup_half8x8 a;
                simdgroup_load(a, tile + ((row + dy) * (TX + 2) + col + dx) * N + k * 8, N);
                for (int j = 0; j < N / 8; j++) simdgroup_multiply_accumulate(acc[s][j], a, w[k][j], acc[s][j]);
            }
        }
    }
    for (int s = 0; s < SEGS; s++) {
        const int seg = sg * SEGS + s, row = seg / (TX / 8), col = (seg % (TX / 8)) * 8;
        for (int j = 0; j < N / 8; j++) simdgroup_store(acc[s][j], products + (row * TX + col) * N + j * 8, N);
    }
#else
    // One block of 8 output channels at a time: all of them at once spills registers at N = 32 (140 against
    // 74 ms for C4F32 at 1080p on an M2 Pro).
    for (int j = 0; j < N / 8; j++) {
        simdgroup_half8x8 acc[SEGS];
        for (int s = 0; s < SEGS; s++) acc[s] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        for (int t = 0; t < 9; t++) {
            const int dx = t % 3, dy = t / 3;
            for (int k = 0; k < N / 8; k++) {
                simdgroup_half8x8 w;
                simdgroup_load(w, weights + t * N * N + k * 8 * N + j * 8, N);
                for (int s = 0; s < SEGS; s++) {
                    const int seg = sg * SEGS + s, row = seg / (TX / 8), col = (seg % (TX / 8)) * 8;
                    simdgroup_half8x8 a;
                    simdgroup_load(a, tile + ((row + dy) * (TX + 2) + col + dx) * N + k * 8, N);
                    simdgroup_multiply_accumulate(acc[s], a, w, acc[s]);
                }
            }
        }
        for (int s = 0; s < SEGS; s++) {
            const int seg = sg * SEGS + s, row = seg / (TX / 8), col = (seg % (TX / 8)) * 8;
            simdgroup_store(acc[s], products + (row * TX + col) * N + j * 8, N);
        }
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *bias = weights + 9 * N * N;
    for (int i = tid; i < TX * TY * G; i += THREADS) {
        const int g = i % G, p = i / G;
        const int2 pix = base + int2(p % TX, p / TX);
        if (pix.x >= int(P.width) || pix.y >= int(P.height)) { continue; }
        half4 v = *(threadgroup half4 *)(products + p * N + g * 4) + *(device const half4 *)(bias + g * 4);
        if (artcnnRelu) { v = max(v, half4(0.0h)); }
        dst.write(v, uint2(pix.x * PX + g % PX, pix.y * 2 + g / PX));
    }
}
"""


def main():
    kernels_by_model = {}
    weights_by_model = {}
    threadgroups = {}
    packings = {}
    license_text = None
    for model, filename in MODELS:
        header, passes = parse(os.path.join(sys.argv[1], filename))
        license_text = license_text or header
        names = [p['desc'].split('(')[-1].rstrip(')') for p in passes]
        assert names == ['Conv2D', 'Conv2D-1-ReLU', 'Conv2D-2-ReLU', 'Conv2D-3-ReLU', 'Conv2D-4-ReLU',
                         'Conv2D-5', 'Conv2D-6', 'Depth-To-Space'], f'{filename}: unexpected passes {names}'
        assert passes[6]['binds'] == ['conv2d', 'conv2d_5'] and passes[7]['binds'] == ['conv2d_6']
        packing = (int(passes[0]['width']), int(passes[0]['height']))
        out = []
        groups = []
        for i, p in enumerate(passes[:7]):
            kernel, tg = translate(p, i, packing)
            out.append(kernel)
            groups.append(tg)
        n = 4 * packing[0] * packing[1]
        weights_by_model[model] = base64.b64encode(hidden_weights(passes, n)).decode()
        out.append(f'#define ARTCNN_N {n}\n#define ARTCNN_PX {packing[0]}\n' + MATRIX_KERNEL)
        kernels_by_model[model] = '\n'.join(out)
        threadgroups[model] = groups
        packings[model] = packing
    notice = '\n'.join('/// ' + l if l else '///' for l in license_text)
    cases = []
    for model, _ in MODELS:
        src = PRELUDE + '\n' + kernels_by_model[model]
        tgs = ', '.join(f'({w}, {h})' for w, h in threadgroups[model])
        b64 = weights_by_model[model]
        lines = '\n'.join('    ' + b64[i:i + 112] for i in range(0, len(b64), 112))
        cases.append(f"""    /// {model.upper()}: features packed {packings[model][0]}x{packings[model][1]} texels per source pixel.
    static let {model}Packing = ({packings[model][0]}, {packings[model][1]})
    /// Conv2D-1...5's weights for `artcnnHiddenMatrix`, float16 little-endian, per pass [tap][in][out] then bias.
    static let {model}HiddenWeights = \"\"\"
{lines}
    \"\"\"
    static let {model}Threadgroups: [(Int, Int)] = [{tgs}]
    static let {model} = #\"\"\"
{src}\"\"\"#
""")
    swift = f"""// Generated by Scripts/generate-artcnn-shader.py from ArtCNN {REVISION}
// GLSL/ArtCNN_C4F16.glsl and GLSL/ArtCNN_C4F32.glsl. Do not edit by hand.

/// The ArtCNN scaler's kernels: ArtCNN's C4F16 and C4F32 luma doublers, one compute kernel per mpv hook pass
/// (`artcnnConv0` to `artcnnConv6`, the threadgroup tiling and weights unchanged), then `artcnnOutput`,
/// which writes the output frame's planes. One source per model, compiled only for the model in use.
/// The original notice:
///
{notice}
enum ArtCNNShaderSource {{
    static let convolutionKernels = (0..<7).map {{ "artcnnConv\\($0)" }}
    static let outputKernel = "artcnnOutput"
    static let hiddenMatrixKernel = "artcnnHiddenMatrix"
    /// Which of Conv2D-1...5 end in a ReLU (`artcnnRelu`).
    static let hiddenReLU = [true, true, true, true, false]

{chr(10).join(cases)}}}
"""
    with open(OUT, 'w') as f:
        f.write(swift)
    print('wrote', os.path.normpath(OUT))


if __name__ == '__main__':
    main()
