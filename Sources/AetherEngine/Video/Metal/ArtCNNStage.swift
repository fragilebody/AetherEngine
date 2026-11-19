#if !targetEnvironment(simulator)
import CoreVideo
import Foundation
import Metal

/// Fork: the ArtCNN scaler (`UpscaleScaler.artCNN`, `ArtCNNShaderSource`): ArtCNN's C4F16 or C4F32 luma doubler
/// as the only stage of a `FrameChain`. It is planar: the network reads the decoder's luma plane and writes the
/// output frame's luma plane at exactly twice the size, and the chroma is scaled bilinearly into the output's
/// chroma plane in the same last pass. There is no RGB texture and no colour conversion, as in mpv, where
/// ArtCNN hooks LUMA.
///
/// Seven passes as the GLSL has them (`artcnnConv0` to `artcnnConv6`, the GLSL's workgroups, one source pixel
/// per thread), then `artcnnOutput`. Features live in rgba16Float textures with the GLSL's packing (C4F16: 2x2
/// texels per source pixel, C4F32: 4x2), three of them: the first layer's output, kept for the long skip into
/// Conv2D-6, and two that the hidden layers alternate between. On a GPU with matrix units (Apple7 and later:
/// M1, A14) the hidden passes Conv2D-1...5, nearly all the arithmetic, run as `artcnnHiddenMatrix` instead.
@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
final class ArtCNNStage: FrameChainStage {
    let model: ArtCNNModel
    /// Whether Conv2D-1...5 run on the matrix units; nil decides by the GPU.
    private let matrixUnitsRequested: Bool?
    /// What `prepare` settled on.
    private(set) var usesMatrixUnits = false
    private var kernels: ArtCNNKernels?
    private var first: MTLTexture?
    private var ping: MTLTexture?
    private var pong: MTLTexture?
    private var last: MTLTexture?
    private var source = (0, 0)

    /// `matrixUnits` false keeps every pass on the generated GLSL kernels (the reference the matrix passes are
    /// tested against); nil uses the matrix units where the GPU has them.
    init(model: ArtCNNModel, matrixUnits: Bool? = nil) {
        self.model = model
        matrixUnitsRequested = matrixUnits
    }

    var isPlanar: Bool { true }

    func prepare(io: MetalFrameIO, kernels _: FrameChainKernels, input: (Int, Int),
                 finalTarget _: (Int, Int)?) throws -> (Int, Int) {
        let kernels = try ArtCNNKernels.shared(for: io.device, model: model)
        let (px, py) = kernels.packing
        let device = io.device
        first = try FrameChain.texture(device, input.0 * px, input.1 * py)
        ping = try FrameChain.texture(device, input.0 * px, input.1 * py)
        pong = try FrameChain.texture(device, input.0 * px, input.1 * py)
        last = try FrameChain.texture(device, input.0, input.1)
        self.kernels = kernels
        usesMatrixUnits = (matrixUnitsRequested ?? true) && kernels.hiddenMatrix != nil
        source = input
        return (input.0 * 2, input.1 * 2)
    }

    /// Unused: the chain runs this stage planar only.
    func encode(_ encoder: MTLComputeCommandEncoder, input: MTLTexture) -> MTLTexture { input }

    func encodePlanar(_ encoder: MTLComputeCommandEncoder, luma: MTLTexture, chroma: MTLTexture,
                      outLuma: MTLTexture, outChroma: MTLTexture, tenBit: Bool, origin: (Int, Int)) {
        guard let kernels, let first, let ping, let pong, let last else { return }
        var params = ArtCNNParams(originX: UInt32(origin.0), originY: UInt32(origin.1),
                                  width: UInt32(source.0), height: UInt32(source.1), tenBit: tenBit)
        encoder.setBytes(&params, length: MemoryLayout<ArtCNNParams>.size, index: 1)
        // conv0: luma -> first; conv1...conv5 alternate ping/pong; conv6 reads first + conv5's output.
        let passes: [(Int, [MTLTexture], MTLTexture)] = [
            (0, [luma], first), (1, [first], ping), (2, [ping], pong), (3, [pong], ping), (4, [ping], pong),
            (5, [pong], ping), (6, [first, ping], last),
        ]
        for (i, inputs, output) in passes {
            if usesMatrixUnits, (1...5).contains(i), let matrix = kernels.hiddenMatrix {
                encoder.setComputePipelineState(ArtCNNShaderSource.hiddenReLU[i - 1] ? matrix.relu : matrix.linear)
                encoder.setTexture(inputs[0], index: 0)
                encoder.setTexture(output, index: 1)
                encoder.setBuffer(matrix.weights, offset: (i - 1) * matrix.passBytes, index: 2)
                // 16x8-pixel tiles, four simdgroups each.
                encoder.dispatchThreadgroups(
                    MTLSize(width: (source.0 + 15) / 16, height: (source.1 + 7) / 8, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                continue
            }
            encoder.setComputePipelineState(kernels.convolutions[i])
            for (t, texture) in inputs.enumerated() { encoder.setTexture(texture, index: t) }
            encoder.setTexture(output, index: inputs.count)
            // Whole threadgroups: each one loads its tile with a halo, every thread taking a share.
            let tg = kernels.threadgroups[i]
            encoder.dispatchThreadgroups(
                MTLSize(width: (source.0 + tg.0 - 1) / tg.0, height: (source.1 + tg.1 - 1) / tg.1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg.0, height: tg.1, depth: 1))
        }
        encoder.setComputePipelineState(kernels.output)
        encoder.setTexture(last, index: 0)
        encoder.setTexture(chroma, index: 1)
        encoder.setTexture(outLuma, index: 2)
        encoder.setTexture(outChroma, index: 3)
        let w = kernels.output.threadExecutionWidth
        let h = max(1, min(16, kernels.output.maxTotalThreadsPerThreadgroup / w))
        encoder.dispatchThreadgroups(
            MTLSize(width: (source.0 + w - 1) / w, height: (source.1 + h - 1) / h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }
}

/// Layout matches `ArtCNNParams` in `ArtCNNShaderSource`.
struct ArtCNNParams {
    var originX: UInt32
    var originY: UInt32
    var width: UInt32
    var height: UInt32
    var lumaMul: Float
    var codeScale: Float
    var maxCode: Float

    init(originX: UInt32, originY: UInt32, width: UInt32, height: UInt32, tenBit: Bool) {
        self.originX = originX
        self.originY = originY
        self.width = width
        self.height = height
        // 10-bit planes hold the code in the high bits of 16, so a unorm16 read is code * 64 / 65535.
        codeScale = tenBit ? 65535.0 / 64.0 : 255
        maxCode = tenBit ? 1023 : 255
        // The network sees code / maxCode, as mpv's LUMA_mul gives it.
        lumaMul = codeScale / maxCode
    }
}

/// One model's pipelines, compiled once per process, device and model (`MetalPipelineCache`). C4F32's source
/// is about 760 KB and takes about 2 s to compile, so each model is compiled only when it is first used, on
/// the upscaler's build queue.
@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
final class ArtCNNKernels: @unchecked Sendable {
    let convolutions: [MTLComputePipelineState]
    let output: MTLComputePipelineState
    let threadgroups: [(Int, Int)]
    let packing: (Int, Int)
    /// Conv2D-1...5 on the matrix units, nil on a GPU without simdgroup matrices (before Apple7).
    let hiddenMatrix: HiddenMatrix?

    struct HiddenMatrix {
        let relu: MTLComputePipelineState
        let linear: MTLComputePipelineState
        /// The five passes' weights and biases, `passBytes` apart.
        let weights: MTLBuffer
        let passBytes: Int
    }

    private static let caches: [ArtCNNModel: MetalPipelineCache<ArtCNNKernels>] = Dictionary(
        uniqueKeysWithValues: ArtCNNModel.allCases.map { ($0, MetalPipelineCache<ArtCNNKernels>()) })

    private init(convolutions: [MTLComputePipelineState], output: MTLComputePipelineState,
                 threadgroups: [(Int, Int)], packing: (Int, Int), hiddenMatrix: HiddenMatrix?) {
        self.hiddenMatrix = hiddenMatrix
        self.convolutions = convolutions
        self.output = output
        self.threadgroups = threadgroups
        self.packing = packing
    }

    static func shared(for device: MTLDevice, model: ArtCNNModel) throws -> ArtCNNKernels {
        try caches[model]!.resolve(device: device) { try compile(device, model: model) }
    }

    private static func compile(_ device: MTLDevice, model: ArtCNNModel) throws -> ArtCNNKernels {
        let source: String, threadgroups: [(Int, Int)], packing: (Int, Int), weights: String
        switch model {
        case .c4f16:
            (source, threadgroups, packing, weights) = (
                ArtCNNShaderSource.c4f16, ArtCNNShaderSource.c4f16Threadgroups, ArtCNNShaderSource.c4f16Packing,
                ArtCNNShaderSource.c4f16HiddenWeights)
        case .c4f32:
            (source, threadgroups, packing, weights) = (
                ArtCNNShaderSource.c4f32, ArtCNNShaderSource.c4f32Threadgroups, ArtCNNShaderSource.c4f32Packing,
                ArtCNNShaderSource.c4f32HiddenWeights)
        }
        let start = DispatchTime.now().uptimeNanoseconds
        let library = try device.makeLibrary(source: source, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw FrameChain.Failure("ArtCNN kernel \(name) missing")
            }
            return try device.makeComputePipelineState(function: function)
        }
        let convolutions = try ArtCNNShaderSource.convolutionKernels.map(pipeline)
        for (pipeline, tg) in zip(convolutions, threadgroups) where pipeline.maxTotalThreadsPerThreadgroup < tg.0 * tg.1 {
            throw FrameChain.Failure("ArtCNN \(model.rawValue): \(tg.0)x\(tg.1) threads exceed the kernel's "
                                     + "\(pipeline.maxTotalThreadsPerThreadgroup) on this GPU")
        }
        var hiddenMatrix: HiddenMatrix?
        if device.supportsFamily(.apple7) {
            func matrix(relu: Bool) throws -> MTLComputePipelineState {
                let values = MTLFunctionConstantValues()
                var v = relu
                values.setConstantValue(&v, type: .bool, index: 0)
                let function = try library.makeFunction(name: ArtCNNShaderSource.hiddenMatrixKernel, constantValues: values)
                return try device.makeComputePipelineState(function: function)
            }
            let n = 4 * packing.0 * packing.1
            let passBytes = (9 * n * n + n) * 2
            guard let blob = Data(base64Encoded: weights, options: .ignoreUnknownCharacters), blob.count == 5 * passBytes,
                  let buffer = blob.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: blob.count) })
            else { throw FrameChain.Failure("ArtCNN \(model.rawValue): hidden weights malformed") }
            hiddenMatrix = HiddenMatrix(relu: try matrix(relu: true), linear: try matrix(relu: false),
                                        weights: buffer, passBytes: passBytes)
        }
        let kernels = ArtCNNKernels(convolutions: convolutions, output: try pipeline(ArtCNNShaderSource.outputKernel),
                                    threadgroups: threadgroups, packing: packing, hiddenMatrix: hiddenMatrix)
        EngineLog.emit(String(format: "[Upscale] ArtCNN %@ kernels compiled in %.0f ms", model.rawValue.uppercased(),
                              Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000), category: .swPlayback)
        return kernels
    }
}
#endif
