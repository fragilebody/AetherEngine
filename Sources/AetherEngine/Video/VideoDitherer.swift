import CoreMedia
import CoreVideo
import Foundation
import Metal

/// How `VideoDitherer` (and `LoadOptions.videoDither`) dithers SDR video, after mpv's options of the same names.
public struct VideoDitherOptions: Sendable, Equatable {

    /// The threshold matrix, mpv's `--dither`.
    public enum Algorithm: String, Sendable, CaseIterable {
        /// A blue-noise matrix, mpv's default: no visible structure.
        case fruit
        /// A 16x16 Bayer matrix: a regular cross-hatch, cheaper to build.
        case ordered
        /// No dither, plain rounding to the output depth: for a pass that only debands.
        case none
    }

    /// The bit depth the picture is dithered to, mpv's `--dither-depth`.
    public enum Depth: Sendable, Equatable {
        /// 10 bits. The system does not report a panel's depth, so this is the depth of the frames handed
        /// to it, which is also where mpv's `auto` ends up on Apple platforms.
        case auto
        /// 8 or 10; anything below 10 is 8, anything above is 10.
        case bits(Int)
    }

    public var algorithm: Algorithm
    public var depth: Depth
    /// The fruit matrix is `2^fruitSizeLog2` on a side (mpv's `--dither-size-fruit`), 2 to 8. Default 6.
    public var fruitSizeLog2: Int
    /// Turn and mirror the matrix from frame to frame (mpv's `--temporal-dither`). Can flicker on LCD panels.
    public var temporal: Bool
    /// Frames each orientation is kept for with `temporal` (mpv's `--temporal-dither-period`).
    public var temporalPeriod: Int

    public init(algorithm: Algorithm = .fruit, depth: Depth = .auto, fruitSizeLog2: Int = 6,
                temporal: Bool = false, temporalPeriod: Int = 1) {
        self.algorithm = algorithm
        self.depth = depth
        self.fruitSizeLog2 = fruitSizeLog2
        self.temporal = temporal
        self.temporalPeriod = temporalPeriod
    }

    /// The depth frames are dithered to: 8 or 10.
    public var resolvedBits: Int {
        switch depth {
        case .auto: 10
        case .bits(let bits): bits >= 10 ? 10 : 8
        }
    }
}

/// How `VideoDitherer` (and `LoadOptions.videoDeband`) removes banding that is already in the video, after
/// mpv's `--deband` options of the same names and with their defaults. Each iteration averages four samples
/// at a random distance and angle, and a component that is within the threshold of that average takes it;
/// grain then covers what is left.
public struct VideoDebandOptions: Sendable, Equatable {
    /// Rounds per pixel, each reaching further with a tighter threshold (`--deband-iterations`), 0 to 16.
    /// Default 1. 0 adds grain only.
    public var iterations: Int
    /// How close a sample must be to the average to be replaced by it (`--deband-threshold`), 0 to 4096.
    /// Default 48. Higher removes more banding and more fine detail.
    public var threshold: Double
    /// The first iteration's radius in source pixels (`--deband-range`), 1 to 64. Default 16.
    public var range: Double
    /// Noise added afterwards (`--deband-grain`), 0 to 4096. Default 32. Black and neutral chroma stay clean.
    public var grain: Double

    public init(iterations: Int = 1, threshold: Double = 48, range: Double = 16, grain: Double = 32) {
        self.iterations = iterations
        self.threshold = threshold
        self.range = range
        self.grain = grain
    }
}

/// mpv's dither as one Metal pass: a decoded SDR frame (bi-planar Y'CbCr, 8 or 10 bit) is converted to
/// R'G'B' at full precision and quantised to 8 or 10 bits through a threshold matrix tiled over the
/// picture, so the rounding becomes fine noise instead of bands. With `deband` the same pass first removes
/// banding the encoder left in the frame, as mpv's `--deband` does, before the conversion.
///
/// The engine runs it on the software path when `LoadOptions.videoDither` or `LoadOptions.videoDeband` is set. A host that takes
/// AVPlayer's frames itself (an `AVPlayerItemVideoOutput`) can run the same pass on them; ask the output
/// for `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` so the pass sees the source's full depth.
///
/// Safe to call from any thread; calls are serialised.
public final class VideoDitherer: @unchecked Sendable {

    public let options: VideoDitherOptions
    /// The deband stage in front of the dither; nil is none.
    public let deband: VideoDebandOptions?

    /// The depth of the frames `dither(_:)` returns: 8 or 10.
    public var outputBits: Int { options.resolvedBits }

    /// The format of the frames `dither(_:)` returns: `kCVPixelFormatType_ARGB2101010LEPacked` at 10 bits,
    /// `kCVPixelFormatType_32BGRA` at 8.
    public var outputPixelFormat: OSType { Self.pixelFormat(bits: outputBits) }

    static func pixelFormat(bits: Int) -> OSType {
        bits >= 10 ? kCVPixelFormatType_ARGB2101010LEPacked : kCVPixelFormatType_32BGRA
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private let lock = NSLock()
    /// The threshold matrix in use. A fruit matrix the process has not built yet is built in the
    /// background; the ordered one stands in until then, so no frame waits for it or goes undithered.
    private var matrix: MTLTexture
    private var matrixSize: Int
    private var matrixIsFruit: Bool
    private var pool: CVPixelBufferPool?
    private var poolSize: (width: Int, height: Int) = (0, 0)
    private var frameIndex = 0
    private var described = false
    private var failureNoted = false

    /// Nil without a Metal device, or when the pass cannot be compiled.
    public convenience init?(options: VideoDitherOptions = VideoDitherOptions(), deband: VideoDebandOptions? = nil) {
        self.init(options: options, deband: deband, synchronousMatrix: false)
    }

    /// `synchronousMatrix` builds a missing fruit matrix before returning (tests).
    init?(options: VideoDitherOptions, deband: VideoDebandOptions? = nil, synchronousMatrix: Bool) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { return nil }
        self.options = options
        self.deband = deband
        let bits = options.resolvedBits

        let descriptor = MTLRenderPipelineDescriptor()
        guard let library = try? device.makeLibrary(source: DitherShaderSource.source, options: nil) else { return nil }
        descriptor.vertexFunction = library.makeFunction(name: "dither_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "dither_fragment")
        descriptor.colorAttachments[0].pixelFormat = bits >= 10 ? .bgr10a2Unorm : .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else { return nil }

        var fruit: [Float]?
        if options.algorithm == .fruit {
            fruit = synchronousMatrix ? DitherMatrix.cachedFruit(sizeLog2: options.fruitSizeLog2)
                                      : DitherMatrix.builtFruit(sizeLog2: options.fruitSizeLog2)
        }
        // `.none` reads no matrix, but the shader still has one bound.
        let values = options.algorithm == .none ? [0] : fruit ?? DitherMatrix.bayer(size: DitherMatrix.orderedSize)
        guard let first = Self.matrixTexture(values, device: device) else { return nil }

        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.textureCache = cache
        self.matrix = first.texture
        self.matrixSize = first.size
        self.matrixIsFruit = fruit != nil

        if options.algorithm == .fruit, fruit == nil {
            let sizeLog2 = options.fruitSizeLog2
            DitherMatrix.buildQueue.async { [weak self] in
                let values = DitherMatrix.cachedFruit(sizeLog2: sizeLog2)
                guard let self, let built = Self.matrixTexture(values, device: device) else { return }
                self.lock.lock()
                self.matrix = built.texture
                self.matrixSize = built.size
                self.matrixIsFruit = true
                self.lock.unlock()
            }
        }
    }

    private static func matrixTexture(_ values: [Float], device: MTLDevice) -> (texture: MTLTexture, size: Int)? {
        let size = Int(Double(values.count).squareRoot().rounded())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: size, height: size, mipmapped: false)
        descriptor.usage = [.shaderRead]
        guard size > 0, size * size == values.count, let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        values.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0,
                            withBytes: raw.baseAddress!, bytesPerRow: size * MemoryLayout<Float>.stride)
        }
        return (texture, size)
    }

    /// `buffer` dithered into a new RGB buffer of the same size (`outputPixelFormat`), carrying its colour
    /// tags, clean aperture and pixel aspect. Nil when the frame is not one this pass takes (HDR or Dolby
    /// Vision transfer, a format that is not 8 or 10 bit bi-planar Y'CbCr) or the GPU failed: show `buffer` then.
    ///
    /// `debandRangeScale` multiplies the deband radius: a caller whose frames were scaled up from the source
    /// passes the factor (frame height over source height) so the radius stays in source pixels.
    public func dither(_ buffer: CVPixelBuffer, debandRangeScale: Double = 1) -> CVPixelBuffer? {
        guard let layout = Self.layout(of: CVPixelBufferGetPixelFormatType(buffer)), Self.isSDR(buffer) else { return nil }
        lock.lock(); defer { lock.unlock() }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0,
              let output = makeOutput(width: width, height: height),
              let luma = planeTexture(buffer, plane: 0, format: layout.bits > 8 ? .r16Unorm : .r8Unorm),
              let chroma = planeTexture(buffer, plane: 1, format: layout.bits > 8 ? .rg16Unorm : .rg8Unorm),
              let target = targetTexture(output),
              let commands = queue.makeCommandBuffer()
        else { return failed("no buffer or texture for a \(width)x\(height) frame") }

        var params = Self.params(layout: layout, matrix: UpscaleMatrix.of(buffer), bits: outputBits,
                                 matrixSize: options.algorithm == .none ? 0 : matrixSize,
                                 chromaOffset: Self.chromaOffset(of: buffer, luma: (width, height),
                                                                 chroma: (chroma.texture.width, chroma.texture.height)),
                                 phase: Self.phase(frame: frameIndex, options: options),
                                 deband: deband, debandRangeScale: debandRangeScale,
                                 frameIndex: UInt32(truncatingIfNeeded: frameIndex))
        frameIndex &+= 1

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target.texture
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return failed("no render encoder") }
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBytes(&params, length: MemoryLayout<DitherShaderParams>.stride, index: 0)
        encoder.setFragmentTexture(luma.texture, index: 0)
        encoder.setFragmentTexture(chroma.texture, index: 1)
        encoder.setFragmentTexture(matrix, index: 2)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        // The CVMetalTextures must live until the GPU is done with them.
        withExtendedLifetime((luma.reference, chroma.reference, target.reference)) {}
        CVMetalTextureCacheFlush(textureCache, 0)
        guard commands.status == .completed else {
            return failed("command buffer \(commands.status.rawValue): \(commands.error.map { "\($0)" } ?? "no error")")
        }

        Self.carryAttachments(from: buffer, to: output)
        if !described {
            described = true
            let matrixName = switch options.algorithm {
            case .none: "no matrix (rounding)"
            case .ordered: "ordered \(matrixSize)x\(matrixSize)"
            case .fruit: matrixIsFruit ? "fruit \(matrixSize)x\(matrixSize)" : "fruit (ordered until its matrix is built)"
            }
            let temporal = options.temporal && options.algorithm != .none ? ", temporal" : ""
            let debanded = deband.map { ", deband (iterations \($0.iterations), threshold \(Self.format($0.threshold)), "
                + "range \(Self.format($0.range)), grain \(Self.format($0.grain)))" } ?? ""
            EngineLog.emit("[Dither] \(matrixName)\(temporal)\(debanded) to \(outputBits) bits, "
                           + "from \(layout.bits)-bit \(width)x\(height)", category: .engine)
        }
        return output
    }

    private static func format(_ value: Double) -> String {
        value.rounded() == value ? String(Int(value)) : String(value)
    }

    private func failed(_ reason: String) -> CVPixelBuffer? {
        if !failureNoted {
            failureNoted = true
            EngineLog.emit("[Dither] frame passed through: \(reason)", category: .engine)
        }
        return nil
    }

    // MARK: - Frame description

    /// What a bi-planar Y'CbCr format holds.
    struct Layout: Equatable {
        var bits: Int
        var videoRange: Bool
    }

    static func layout(of format: OSType) -> Layout? {
        switch format {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_422YpCbCr8BiPlanarVideoRange,
             kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange:
            Layout(bits: 8, videoRange: true)
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_422YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_444YpCbCr8BiPlanarFullRange:
            Layout(bits: 8, videoRange: false)
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
             kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange:
            Layout(bits: 10, videoRange: true)
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange, kCVPixelFormatType_422YpCbCr10BiPlanarFullRange,
             kCVPixelFormatType_444YpCbCr10BiPlanarFullRange:
            Layout(bits: 10, videoRange: false)
        default:
            nil
        }
    }

    /// False for PQ and HLG: this pass has no tone mapping, and those frames keep the system's path.
    static func isSDR(_ buffer: CVPixelBuffer) -> Bool {
        guard let transfer = CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil) as? String else { return true }
        return transfer != kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String
            && transfer != kCVImageBufferTransferFunction_ITU_R_2100_HLG as String
    }

    /// The shader's constants for one frame. A 10-bit plane holds its code in the high bits of 16, so a
    /// sample of 1.0 is code 65535 / 64, not 1023. `matrixSize` 0 is no matrix (plain rounding).
    static func params(layout: Layout, matrix: UpscaleMatrix, bits: Int, matrixSize: Int,
                       chromaOffset: SIMD2<Float>, phase: UInt32, deband: VideoDebandOptions? = nil,
                       debandRangeScale: Double = 1, frameIndex: UInt32 = 0) -> DitherShaderParams {
        let codeScale: Float = layout.bits > 8 ? 65535.0 / 64.0 : 255.0
        let unit: Float = layout.bits > 8 ? 4 : 1
        let maxCode: Float = layout.bits > 8 ? 1023 : 255
        let yScale, yOffset, cScale, cOffset: Float
        if layout.videoRange {
            yScale = codeScale / (219 * unit)
            yOffset = -16 / 219
            cScale = codeScale / (224 * unit)
            cOffset = -128 / 224
        } else {
            yScale = codeScale / maxCode
            yOffset = 0
            cScale = codeScale / maxCode
            cOffset = -128 * unit / maxCode
        }
        // libplacebo takes the threshold and grain as fractions of the plane's code range (code / maxCode);
        // here a unit of Y' is 219 codes (224 for chroma) in video range, so they are scaled to match.
        let lumaUnit: Float = layout.videoRange ? maxCode / (219 * unit) : 1
        let chromaUnit: Float = layout.videoRange ? maxCode / (224 * unit) : 1
        var debandThreshold = SIMD2<Float>.zero, debandGrain = SIMD2<Float>.zero
        var debandIterations: UInt32 = 0, debandRange: Float = 0
        if let deband {
            // mpv's option scales, as vo=gpu-next hands them to libplacebo (which divides by 1000 again).
            let threshold = Float(min(max(deband.threshold, 0), 4096) / 16.384 / 1000)
            let grain = Float(min(max(deband.grain, 0), 4096) / 8.192 / 1000)
            debandThreshold = SIMD2(threshold * lumaUnit, threshold * chromaUnit)
            debandGrain = SIMD2(grain * lumaUnit, grain * chromaUnit)
            debandIterations = UInt32(min(max(deband.iterations, 0), 16))
            debandRange = Float(min(max(deband.range, 1), 64) * max(debandRangeScale, 1))
        }
        return DitherShaderParams(
            yScale: yScale, yOffset: yOffset, cScale: cScale, cOffset: cOffset,
            kr: matrix.kr, kb: matrix.kb,
            steps: Float((1 << (bits >= 10 ? 10 : 8)) - 1),
            biasCentre: matrixSize > 0 ? 0.5 / Float(matrixSize * matrixSize) : 0.5,
            chromaOffset: chromaOffset, debandThreshold: debandThreshold, debandGrain: debandGrain,
            matrixSize: UInt32(matrixSize), phase: phase,
            debandIterations: debandIterations, frameIndex: frameIndex, debandRange: debandRange)
    }

    /// Where the chroma samples sit, in chroma texels to add to a centred lookup. Untagged is left-sited,
    /// what H.264, HEVC and AV1 4:2:0 use by default. No offset along an axis that is not subsampled.
    static func chromaOffset(of buffer: CVPixelBuffer, luma: (Int, Int), chroma: (Int, Int)) -> SIMD2<Float> {
        let location = CVBufferCopyAttachment(buffer, kCVImageBufferChromaLocationTopFieldKey, nil) as? String
        let horizontal = luma.0 > 0 ? 0.5 * (1 - Float(chroma.0) / Float(luma.0)) : 0
        let vertical = luma.1 > 0 ? 0.5 * (1 - Float(chroma.1) / Float(luma.1)) : 0
        func at(_ key: CFString) -> Bool { location == key as String }
        if at(kCVImageBufferChromaLocation_Center) { return SIMD2(0, 0) }
        if at(kCVImageBufferChromaLocation_TopLeft) { return SIMD2(horizontal, vertical) }
        if at(kCVImageBufferChromaLocation_Top) { return SIMD2(0, vertical) }
        if at(kCVImageBufferChromaLocation_BottomLeft) { return SIMD2(horizontal, -vertical) }
        if at(kCVImageBufferChromaLocation_Bottom) { return SIMD2(0, -vertical) }
        return SIMD2(horizontal, 0)
    }

    /// mpv's temporal dither: eight orientations of the matrix, one per `temporalPeriod` frames.
    static func phase(frame: Int, options: VideoDitherOptions) -> UInt32 {
        guard options.temporal, options.algorithm != .none else { return 0 }
        return UInt32((frame / max(1, options.temporalPeriod)) % 8)
    }

    /// Tags that describe Y'CbCr samples, which the RGB output no longer has.
    static let droppedAttachmentKeys: Set<String> = [
        kCVImageBufferYCbCrMatrixKey as String,
        kCVImageBufferChromaLocationTopFieldKey as String, kCVImageBufferChromaLocationBottomFieldKey as String,
        kCVImageBufferChromaSubsamplingKey as String,
    ]

    /// The output keeps the input's colour primaries and transfer (the values are the same R'G'B'), its
    /// clean aperture and pixel aspect. An untagged input is BT.709, as it is decoded.
    static func carryAttachments(from source: CVPixelBuffer, to output: CVPixelBuffer) {
        CVBufferRemoveAllAttachments(output)
        var carried = (CVBufferCopyAttachments(source, .shouldPropagate) as? [String: Any]) ?? [:]
        for key in droppedAttachmentKeys { carried[key] = nil }
        if carried[kCVImageBufferColorPrimariesKey as String] == nil {
            carried[kCVImageBufferColorPrimariesKey as String] = kCVImageBufferColorPrimaries_ITU_R_709_2
        }
        if carried[kCVImageBufferTransferFunctionKey as String] == nil {
            carried[kCVImageBufferTransferFunctionKey as String] = kCVImageBufferTransferFunction_ITU_R_709_2
        }
        CVBufferSetAttachments(output, carried as CFDictionary, .shouldPropagate)
    }

    // MARK: - Buffers

    private struct CacheTexture {
        let reference: CVMetalTexture
        let texture: MTLTexture
    }

    private func planeTexture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) -> CacheTexture? {
        var reference: CVMetalTexture?
        guard CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, nil, format,
            CVPixelBufferGetWidthOfPlane(buffer, plane), CVPixelBufferGetHeightOfPlane(buffer, plane), plane,
            &reference) == kCVReturnSuccess,
              let reference, let texture = CVMetalTextureGetTexture(reference) else { return nil }
        return CacheTexture(reference: reference, texture: texture)
    }

    private func targetTexture(_ buffer: CVPixelBuffer) -> CacheTexture? {
        var reference: CVMetalTexture?
        let usage: MTLTextureUsage = [.renderTarget, .shaderRead]
        let attributes = [kCVMetalTextureUsage as String: usage.rawValue] as CFDictionary
        guard CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, buffer, attributes, outputBits >= 10 ? .bgr10a2Unorm : .bgra8Unorm,
            CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer), 0, &reference) == kCVReturnSuccess,
              let reference, let texture = CVMetalTextureGetTexture(reference) else { return nil }
        return CacheTexture(reference: reference, texture: texture)
    }

    private func makeOutput(width: Int, height: Int) -> CVPixelBuffer? {
        if pool == nil || poolSize != (width, height) {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: outputPixelFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &created)
            pool = created
            poolSize = (width, height)
        }
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        return buffer
    }
}
