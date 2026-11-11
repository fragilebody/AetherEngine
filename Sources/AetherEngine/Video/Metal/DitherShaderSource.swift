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
    var matrixSize: UInt32
    /// Temporal dithering: 0...7, the matrix turned in quarter turns (0...3) and mirrored (4...7).
    var phase: UInt32
}

/// One pass: bi-planar Y'CbCr to R'G'B' at full precision, then mpv's dither to the output's bit depth.
/// SDR only, so the conversion is the plain matrix on the coded (gamma) values, where mpv dithers too.
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
        uint matrixSize;
        uint phase;
    };

    struct DitherVertex { float4 position [[position]]; };

    // One triangle that covers the target.
    vertex DitherVertex dither_vertex(uint id [[vertex_id]]) {
        DitherVertex out;
        out.position = float4(float((id << 1) & 2) * 2.0 - 1.0, float(id & 2) * 2.0 - 1.0, 0.0, 1.0);
        return out;
    }

    fragment float4 dither_fragment(DitherVertex in [[stage_in]],
                                    constant DitherParams &p [[buffer(0)]],
                                    texture2d<float> yTex [[texture(0)]],
                                    texture2d<float> cTex [[texture(1)]],
                                    texture2d<float> matrix [[texture(2)]]) {
        constexpr sampler chromaSampler(coord::normalized, filter::linear, address::clamp_to_edge);
        uint2 pos = uint2(in.position.xy);
        float2 lumaSize = float2(yTex.get_width(), yTex.get_height());
        float2 chromaSize = float2(cTex.get_width(), cTex.get_height());

        float y = yTex.read(pos).r * p.yScale + p.yOffset;
        float2 uv = in.position.xy / lumaSize + p.chromaOffset / chromaSize;
        float2 c = cTex.sample(chromaSampler, uv).rg * p.cScale + p.cOffset;

        float kg = 1.0 - p.kr - p.kb;
        float r = y + 2.0 * (1.0 - p.kr) * c.y;
        float b = y + 2.0 * (1.0 - p.kb) * c.x;
        float g = (y - p.kr * r - p.kb * b) / kg;
        float3 rgb = saturate(float3(r, g, b));
        // Float error must not let the pattern lift true black off zero.
        rgb = select(rgb, float3(0.0), rgb < float3(1e-5));

        uint n = p.matrixSize;
        uint2 m = pos % n;
        if (p.phase >= 4) { m.x = n - 1 - m.x; }
        switch (p.phase & 3) {
            case 1: m = uint2(n - 1 - m.y, m.x); break;
            case 2: m = uint2(n - 1 - m.x, n - 1 - m.y); break;
            case 3: m = uint2(m.y, n - 1 - m.x); break;
            default: break;
        }
        float bias = matrix.read(m).r + p.biasCentre;
        rgb = floor(rgb * p.steps + bias) / p.steps;
        return float4(rgb, 1.0);
    }
    """
}
