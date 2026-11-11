import Foundation

/// Layout matches `DitherParams` in the shader.
struct DitherShaderParams {
    /// Plane sample (0...1) to Y': `y = sample * yScale + yOffset`.
    var yScale: Float
    var yOffset: Float
    /// Plane sample (0...1) to Cb / Cr around zero.
    var cScale: Float
    var cOffset: Float
    var kr: Float
    var kb: Float
    /// `2^depth - 1`: the output's code steps.
    var steps: Float
    /// Half a matrix rank, so the threshold's mean is exactly 0.5 (plain rounding's bias).
    var biasCentre: Float
    /// Where chroma samples sit relative to luma, in chroma texels.
    var chromaOffset: SIMD2<Float>
    /// Deband threshold and grain for (Y', chroma), in the normalised units of `yScale` / `cScale`.
    /// Zero grain adds none.
    var debandThreshold: SIMD2<Float> = .zero
    var debandGrain: SIMD2<Float> = .zero
    /// 0 is no matrix: plain rounding (`VideoDitherOptions.Algorithm.none`).
    var matrixSize: UInt32
    /// Temporal dithering: 0...7, the matrix turned in quarter turns (0...3) and mirrored (4...7).
    var phase: UInt32
    /// Deband iterations; 0 skips the deband loop.
    var debandIterations: UInt32 = 0
    /// Seeds the deband hash with the pixel, so the grain changes from frame to frame.
    var frameIndex: UInt32 = 0
    /// Deband radius of the first iteration, in luma pixels of the frame processed.
    var debandRange: Float = 0
}

/// One pass: bi-planar Y'CbCr, optionally debanded (libplacebo's `pl_shader_deband`, which mpv's `--deband`
/// runs), to R'G'B' at full precision, then mpv's dither to the output's bit depth. SDR only, so the
/// conversion is the plain matrix on the coded (gamma) values, where mpv dithers too.
enum DitherShaderSource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct DitherParams {
        float yScale; float yOffset;
        float cScale; float cOffset;
        float kr; float kb;
        float steps; float biasCentre;
        float2 chromaOffset;
        float2 debandThreshold;
        float2 debandGrain;
        uint matrixSize;
        uint phase;
        uint debandIterations;
        uint frameIndex;
        float debandRange;
    };

    struct DitherVertex { float4 position [[position]]; };

    // One triangle that covers the target.
    vertex DitherVertex dither_vertex(uint id [[vertex_id]]) {
        DitherVertex out;
        out.position = float4(float((id << 1) & 2) * 2.0 - 1.0, float(id & 2) * 2.0 - 1.0, 0.0, 1.0);
        return out;
    }

    // libplacebo's PRNG: each call advances the state.
    static uint3 pcg3d(uint3 v) {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
        v ^= v >> 16u;
        v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
        return v;
    }

    static float3 prng(thread uint3 &state) {
        state = pcg3d(state);
        return float3(state) * (1.0 / 4294967295.0);
    }

    // Normalised (Y', Cb, Cr) at a position in luma pixels, clamped to the frame. Chroma sits around 0.
    static float3 readYCC(float2 position, constant DitherParams &p,
                          texture2d<float> yTex, texture2d<float> cTex, float2 lumaSize, float2 chromaSize) {
        constexpr sampler chromaSampler(coord::normalized, filter::linear, address::clamp_to_edge);
        float2 q = clamp(position, float2(0.0), lumaSize - 0.5);
        float y = yTex.read(uint2(q)).r * p.yScale + p.yOffset;
        float2 uv = q / lumaSize + p.chromaOffset / chromaSize;
        float2 c = cTex.sample(chromaSampler, uv).rg * p.cScale + p.cOffset;
        return float3(y, c);
    }

    fragment float4 dither_fragment(DitherVertex in [[stage_in]],
                                    constant DitherParams &p [[buffer(0)]],
                                    texture2d<float> yTex [[texture(0)]],
                                    texture2d<float> cTex [[texture(1)]],
                                    texture2d<float> matrix [[texture(2)]]) {
        uint2 pos = uint2(in.position.xy);
        float2 lumaSize = float2(yTex.get_width(), yTex.get_height());
        float2 chromaSize = float2(cTex.get_width(), cTex.get_height());

        float3 ycc = readYCC(in.position.xy, p, yTex, cTex, lumaSize, chromaSize);

        // Deband: a component close enough to the average of four samples at a random offset (turned in
        // quarter turns) takes that average, each iteration reaching further with a tighter threshold.
        uint3 state = uint3(pos, p.frameIndex);
        float3 threshold = float3(p.debandThreshold.x, p.debandThreshold.y, p.debandThreshold.y);
        for (uint i = 1; i <= p.debandIterations; i++) {
            float2 d = prng(state).xy * float2(float(i) * p.debandRange, 2.0 * M_PI_F);
            d = d.x * float2(cos(d.y), sin(d.y));
            float3 avg = readYCC(in.position.xy + float2( d.x,  d.y), p, yTex, cTex, lumaSize, chromaSize)
                       + readYCC(in.position.xy + float2(-d.x,  d.y), p, yTex, cTex, lumaSize, chromaSize)
                       + readYCC(in.position.xy + float2(-d.x, -d.y), p, yTex, cTex, lumaSize, chromaSize)
                       + readYCC(in.position.xy + float2( d.x, -d.y), p, yTex, cTex, lumaSize, chromaSize);
            avg *= 0.25;
            ycc = select(avg, ycc, abs(ycc - avg) > threshold / float(i));
        }
        // Grain, scaled down near black and neutral chroma (both 0 here) so those stay clean.
        if (p.debandGrain.x > 0.0 || p.debandGrain.y > 0.0) {
            float3 grain = float3(p.debandGrain.x, p.debandGrain.y, p.debandGrain.y);
            ycc += min(abs(ycc), grain) * (prng(state) - 0.5);
        }

        float y = ycc.x;
        float2 c = ycc.yz;
        float kg = 1.0 - p.kr - p.kb;
        float r = y + 2.0 * (1.0 - p.kr) * c.y;
        float b = y + 2.0 * (1.0 - p.kb) * c.x;
        float g = (y - p.kr * r - p.kb * b) / kg;
        float3 rgb = saturate(float3(r, g, b));
        // Float error must not let the pattern lift true black off zero.
        rgb = select(rgb, float3(0.0), rgb < float3(1e-5));

        float bias = p.biasCentre;
        uint n = p.matrixSize;
        if (n > 0) {
            uint2 m = pos % n;
            if (p.phase >= 4) { m.x = n - 1 - m.x; }
            switch (p.phase & 3) {
                case 1: m = uint2(n - 1 - m.y, m.x); break;
                case 2: m = uint2(n - 1 - m.x, n - 1 - m.y); break;
                case 3: m = uint2(m.y, n - 1 - m.x); break;
                default: break;
            }
            bias += matrix.read(m).r;
        }
        rgb = floor(rgb * p.steps + bias) / p.steps;
        return float4(rgb, 1.0);
    }
    """
}
