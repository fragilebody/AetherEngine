import Testing
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import Metal
@testable import AetherEngine

/// Fork: the ArtCNN scaler against ArtCNN's own ONNX models (`ArtCNNReferenceData`, onnxruntime in float32).
/// The kernels run in half precision and repeat the edge sample where ONNX pads with zeros, so the interior
/// (more than 7 source pixels from the edge, the network's reach) is compared within rounding, and the
/// border only loosely.
@Suite("ArtCNN stage", .serialized)
struct ArtCNNStageTests {

    private static let w = ArtCNNReferenceData.width
    private static let h = ArtCNNReferenceData.height
    /// Seven 3x3 layers: an output sample further than this from the edge sees no padding.
    private static let reach = 7 * 2

    private static func decode(_ base64: String) -> Data {
        Data(base64Encoded: base64, options: .ignoreUnknownCharacters)!
    }

    private static func codes8(_ base64: String) -> [Int] { decode(base64).map(Int.init) }

    private static func codes10(_ base64: String) -> [Int] {
        let data = decode(base64)
        return stride(from: 0, to: data.count, by: 2).map { Int(data[$0]) | Int(data[$0 + 1]) << 8 }
    }

    // MARK: - Frames

    struct Failure: Error {}

    /// A 4:2:0 frame of `bw` x `bh` whose luma at (x, y) inside the picture at `origin` is `luma(x, y)` (codes),
    /// outside it 0, and whose chroma is `chroma(x, y)` in chroma samples of the whole buffer.
    private static func frame(_ bw: Int, _ bh: Int, tenBit: Bool, origin: (Int, Int) = (0, 0),
                              luma: (Int, Int) -> Int, chroma: (Int, Int) -> (Int, Int)) throws -> CVPixelBuffer {
        let format = tenBit ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true,
                                      kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()]
        guard CVPixelBufferCreate(kCFAllocatorDefault, bw, bh, format, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { throw Failure() }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        for plane in 0..<2 {
            let base = CVPixelBufferGetBaseAddressOfPlane(pb, plane)!
            let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane)
            let pw = CVPixelBufferGetWidthOfPlane(pb, plane), ph = CVPixelBufferGetHeightOfPlane(pb, plane)
            for row in 0..<ph {
                for x in 0..<pw {
                    let values: [Int]
                    if plane == 0 {
                        let px = x - origin.0, py = row - origin.1
                        values = px >= 0 && py >= 0 && px < w && py < h ? [luma(px, py)] : [0]
                    } else {
                        let c = chroma(x, row)
                        values = [c.0, c.1]
                    }
                    for (i, v) in values.enumerated() {
                        let k = x * values.count + i
                        if tenBit {
                            base.storeBytes(of: UInt16(v << 6).littleEndian, toByteOffset: row * stride + k * 2, as: UInt16.self)
                        } else {
                            base.storeBytes(of: UInt8(v), toByteOffset: row * stride + k, as: UInt8.self)
                        }
                    }
                }
            }
        }
        return pb
    }

    /// The codes of one plane, `channel` of `channels` per sample.
    private static func read(_ pb: CVPixelBuffer, plane: Int, channel: Int = 0) -> (codes: [Int], width: Int) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let tenBit = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        let base = CVPixelBufferGetBaseAddressOfPlane(pb, plane)!
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane)
        let pw = CVPixelBufferGetWidthOfPlane(pb, plane), ph = CVPixelBufferGetHeightOfPlane(pb, plane)
        let channels = plane == 0 ? 1 : 2
        var codes: [Int] = []
        codes.reserveCapacity(pw * ph)
        for row in 0..<ph {
            for x in 0..<pw {
                let k = x * channels + channel
                codes.append(tenBit
                    ? Int(UInt16(littleEndian: base.load(fromByteOffset: row * stride + k * 2, as: UInt16.self)) >> 6)
                    : Int(base.load(fromByteOffset: row * stride + k, as: UInt8.self)))
            }
        }
        return (codes, pw)
    }

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    private static func chain(_ model: ArtCNNModel) throws -> FrameChain {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw Failure() }
        return try FrameChain(device: device, queue: queue, sourceWidth: w, sourceHeight: h,
                              targetWidth: w * 2, targetHeight: h * 2, stages: [ArtCNNStage(model: model)])
    }

    /// Max and mean absolute difference over the interior, and max over the border.
    private static func compare(_ got: [Int], _ want: [Int]) -> (max: Int, mean: Double, border: Int) {
        let ow = w * 2, oh = h * 2
        var maxInterior = 0, maxBorder = 0, sum = 0, n = 0
        for y in 0..<oh {
            for x in 0..<ow {
                let d = abs(got[y * ow + x] - want[y * ow + x])
                if x >= reach && y >= reach && x < ow - reach && y < oh - reach {
                    maxInterior = max(maxInterior, d); sum += d; n += 1
                } else {
                    maxBorder = max(maxBorder, d)
                }
            }
        }
        return (maxInterior, Double(sum) / Double(n), maxBorder)
    }

    // MARK: - Against the ONNX reference

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test(arguments: [ArtCNNModel.c4f16, .c4f32])
    func eightBitMatchesTheONNXModel(model: ArtCNNModel) throws {
        let input = Self.codes8(ArtCNNReferenceData.input8)
        let want = Self.codes8(model == .c4f16 ? ArtCNNReferenceData.c4f16Output8 : ArtCNNReferenceData.c4f32Output8)
        let pb = try Self.frame(Self.w, Self.h, tenBit: false, luma: { input[$1 * Self.w + $0] }, chroma: { _, _ in (128, 128) })
        let chain = try Self.chain(model)
        #expect(chain.isPlanarOnly)
        let (out, _) = try chain.process(pb, picture: .init(x: 0, y: 0, width: Self.w, height: Self.h))
        #expect(CVPixelBufferGetWidth(out) == Self.w * 2 && CVPixelBufferGetHeight(out) == Self.h * 2)
        let diff = Self.compare(Self.read(out, plane: 0).codes, want)
        print("[ArtCNN test] \(model.rawValue) 8-bit vs ONNX: interior max \(diff.max), mean "
              + String(format: "%.3f", diff.mean) + ", border max \(diff.border)")
        // Half-precision accumulation against float32: within a code or two.
        #expect(diff.max <= 2, "interior max \(diff.max), mean \(diff.mean)")
        #expect(diff.mean < 0.35, "interior mean \(diff.mean)")
        // Edge replication against zero padding: only near the border, and bounded.
        #expect(diff.border < 128, "border max \(diff.border)")
    }

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test func tenBitMatchesTheONNXModelAndStaysTenBit() throws {
        let input = Self.codes10(ArtCNNReferenceData.input10)
        let want = Self.codes10(ArtCNNReferenceData.c4f16Output10)
        let pb = try Self.frame(Self.w, Self.h, tenBit: true, luma: { input[$1 * Self.w + $0] }, chroma: { _, _ in (512, 512) })
        let (out, _) = try Self.chain(.c4f16).process(pb, picture: .init(x: 0, y: 0, width: Self.w, height: Self.h))
        #expect(CVPixelBufferGetPixelFormatType(out) == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
        let diff = Self.compare(Self.read(out, plane: 0).codes, want)
        print("[ArtCNN test] c4f16 10-bit vs ONNX: interior max \(diff.max), mean "
              + String(format: "%.3f", diff.mean) + ", border max \(diff.border)")
        #expect(diff.max <= 6, "interior max \(diff.max) (10-bit codes), mean \(diff.mean)")
        #expect(diff.mean < 1.2, "interior mean \(diff.mean)")
    }

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test(arguments: [ArtCNNModel.c4f16, .c4f32])
    func matrixUnitsMatchThePerPixelKernels(model: ArtCNNModel) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple7),
              let queue = device.makeCommandQueue() else { return }
        let input = Self.codes8(ArtCNNReferenceData.input8)
        let pb = try Self.frame(Self.w, Self.h, tenBit: false, luma: { input[$1 * Self.w + $0] }, chroma: { _, _ in (128, 128) })
        func run(_ matrix: Bool) throws -> [Int] {
            let stage = ArtCNNStage(model: model, matrixUnits: matrix)
            let chain = try FrameChain(device: device, queue: queue, sourceWidth: Self.w, sourceHeight: Self.h,
                                       targetWidth: Self.w * 2, targetHeight: Self.h * 2, stages: [stage])
            #expect(stage.usesMatrixUnits == matrix)
            return Self.read(try chain.process(pb, picture: .init(x: 0, y: 0, width: Self.w, height: Self.h)).0, plane: 0).codes
        }
        let a = try run(true), b = try run(false)
        let worst = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        print("[ArtCNN test] \(model.rawValue) matrix vs per-pixel: max \(worst)")
        // Each path is within 2 codes of ONNX (the tests above), so they are within 4 of each other.
        #expect(worst <= 4, "the two orders of the same half-precision sums differ by rounding only")
    }

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test func aPictureInsideALargerBufferGivesTheSameOutput() throws {
        let input = Self.codes8(ArtCNNReferenceData.input8)
        let plain = try Self.frame(Self.w, Self.h, tenBit: false, luma: { input[$1 * Self.w + $0] }, chroma: { _, _ in (128, 128) })
        let framed = try Self.frame(Self.w + 8, Self.h + 6, tenBit: false, origin: (4, 2),
                                    luma: { input[$1 * Self.w + $0] }, chroma: { _, _ in (128, 128) })
        let chain = try Self.chain(.c4f16)
        let (a, _) = try chain.process(plain, picture: .init(x: 0, y: 0, width: Self.w, height: Self.h))
        let (b, _) = try chain.process(framed, picture: .init(x: 4, y: 2, width: Self.w, height: Self.h))
        #expect(Self.read(a, plane: 0).codes == Self.read(b, plane: 0).codes)
    }

    // MARK: - Chroma

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test func chromaIsTheSourceChromaScaledBilinearly() throws {
        // Cb a horizontal ramp, Cr a vertical one, over the 50x35 source chroma plane.
        let pb = try Self.frame(Self.w, Self.h, tenBit: false, luma: { _, _ in 120 },
                                chroma: { x, y in (60 + 2 * x, 70 + 3 * y) })
        let (out, _) = try Self.chain(.c4f16).process(pb, picture: .init(x: 0, y: 0, width: Self.w, height: Self.h))
        let cb = Self.read(out, plane: 1, channel: 0), cr = Self.read(out, plane: 1, channel: 1)
        #expect(cb.width == Self.w)
        // Output chroma sample (x, y) sits at the centre of source luma pixel (x, y): chroma coordinate
        // (x + 0.5) / 2 - 0.5, linear between neighbours. Inside the ramp that is exact.
        for y in 2..<(Self.h - 2) {
            for x in 2..<(Self.w - 2) {
                let wantCb = 60 + 2 * ((Double(x) + 0.5) / 2 - 0.5)
                let wantCr = 70 + 3 * ((Double(y) + 0.5) / 2 - 0.5)
                #expect(abs(Double(cb.codes[y * Self.w + x]) - wantCb) <= 0.51, "Cb at \(x),\(y)")
                #expect(abs(Double(cr.codes[y * Self.w + x]) - wantCr) <= 0.51, "Cr at \(x),\(y)")
            }
        }
    }

    // MARK: - The overrun ladder

    /// A frame period no GPU meets: C4F32 steps down to C4F16 after one budget window, which then switches off.
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    @Test func overrunStepsC4F32DownToC4F16ThenOff() throws {
        ArtCNNCostMemory.reset()
        defer { ArtCNNCostMemory.reset() }
        let source = SourceGeometry(width: Self.w, height: Self.h, sarNum: 1, sarDen: 1)
        let plan = try #require(UpscaleRouteDecision.makePlan(
            source: source, display: CGSize(width: 1920, height: 1080), mlScaleFactors: { _, _ in [] },
            scaler: .artCNN(.c4f32)))
        let upscaler = try #require(FrameUpscaler(plan: plan, frameInterval: 0.0001))
        let pb = try Self.frame(Self.w, Self.h, tenBit: false, luma: { x, y in (x * 7 + y * 3) % 220 + 16 },
                                chroma: { _, _ in (128, 128) })
        var pts = 0.0
        func run(_ frames: Int) {
            for _ in 0..<frames {
                _ = upscaler.upscale(pb, pts: CMTime(seconds: pts, preferredTimescale: 1_000_000))
                pts += 0.0001
            }
        }
        run(FrameUpscaler.budgetWindow + 2)
        #expect(!upscaler.isDisabled)
        #expect(upscaler.isDegraded)
        #expect(upscaler.plan.scaler == .artCNN(.c4f16))
        #expect(ArtCNNCostMemory.overrun(covering: plan, frameInterval: 0.0001, model: .c4f32) != nil)
        run(FrameUpscaler.budgetWindow + 2)
        #expect(upscaler.isDisabled)
        #expect(ArtCNNCostMemory.overrun(covering: plan, frameInterval: 0.0001, model: .c4f16) != nil)
    }
}

/// Fork: what `.artCNN` changes about the upscale route.
@Suite("ArtCNN upscale route")
struct ArtCNNUpscaleRouteTests {

    private func decide(
        model: ArtCNNModel = .c4f16,
        source: SourceGeometry = SourceGeometry(width: 1920, height: 1080, sarNum: 1, sarDen: 1),
        sdr: Bool = true, dv: Bool = false, eightBit: Bool = true, tenBit: Bool = false,
        display: CGSize = CGSize(width: 3840, height: 2160)
    ) -> UpscaleRouteDecision {
        UpscaleRouteDecision.decide(
            preference: true, osSupportsStage: true, metalFXSupported: false,
            source: source, isSDR: sdr, isDolbyVision: dv, isEightBit420: eightBit, display: display,
            wirelessAirPlayRoute: false, demuxedAudioLive: false, mlScaleFactors: { _, _ in [] },
            scaler: .artCNN(model), isTenBit420: tenBit)
    }

    @Test func fullHDOnUHDIsExactlyTwiceTheSource() throws {
        let decision = decide()
        let plan = try #require(decision.plan)
        #expect(plan.scaler == .artCNN(.c4f16) && plan.mlScale == nil && !plan.needsMetalFX)
        #expect(plan.targetWidth == 3840 && plan.targetHeight == 2160)
        #expect(decision.logLine == "[Upscale] taken: 1920x1080 -> 3840x2160 (ArtCNN C4F16 2x)")
    }

    @Test func aLargerDisplayStillGetsExactlyTwiceTheSource() throws {
        let plan = try #require(decide(display: CGSize(width: 6400, height: 3600)).plan)
        #expect(plan.targetWidth == 3840 && plan.targetHeight == 2160)
        #expect(plan.displayWidth == 6400)
    }

    @Test func anamorphicSourceKeepsItsPixelAspect() throws {
        let plan = try #require(decide(model: .c4f32, source: SourceGeometry(width: 720, height: 480, sarNum: 32, sarDen: 27)).plan)
        #expect(plan.targetWidth == 1440 && plan.targetHeight == 960)
        #expect(plan.sarNum == 32 && plan.sarDen == 27)
    }

    @Test func tenBitSDRIsTaken() {
        #expect(decide(eightBit: false, tenBit: true).plan != nil)
    }

    @Test func largerThan1080pHDRAndDolbyVisionAreNot() {
        #expect(decide(source: SourceGeometry(width: 3840, height: 2160, sarNum: 1, sarDen: 1)) == .notTaken(.largerThan1080p))
        #expect(decide(sdr: false) == .notTaken(.hdrOrDolbyVision))
        #expect(decide(dv: true) == .notTaken(.hdrOrDolbyVision))
        #expect(decide(eightBit: false, tenBit: false) == .notTaken(.notFourTwoZero))
    }

    @Test func theDisplayHasToBeMoreThan1_3xTheSource() {
        #expect(decide(display: CGSize(width: 2560, height: 1440)).plan != nil)  // 1.33x
        #expect(decide(display: CGSize(width: 2400, height: 1350)) == .notTaken(.displayUnderArtCNNThreshold))  // 1.25x
        #expect(decide(source: SourceGeometry(width: 1280, height: 720, sarNum: 1, sarDen: 1),
                       display: CGSize(width: 1920, height: 1080)).plan != nil)  // 1.5x
    }

    @Test func plansOfTheTwoModelsAreDistinct() throws {
        let a = try #require(decide(model: .c4f16).plan), b = try #require(decide(model: .c4f32).plan)
        #expect(a != b)
    }
}
