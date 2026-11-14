import Foundation
import Testing
import CoreMedia
import CoreVideo
import Metal
import VideoToolbox
@testable import AetherEngine

@Suite("Dither matrices")
struct DitherMatrixTests {

    private func ranks(_ matrix: [Float]) -> [Int] {
        matrix.map { Int(($0 * Float(matrix.count)).rounded()) }.sorted()
    }

    @Test(arguments: [2, 4, 6])
    func fruitRanksEveryCellOnce(_ sizeLog2: Int) {
        let matrix = DitherMatrix.fruit(sizeLog2: sizeLog2)
        let side = 1 << sizeLog2
        #expect(matrix.count == side * side)
        #expect(ranks(matrix) == Array(0..<(side * side)))
    }

    @Test func fruitIsDeterministicAndCached() {
        #expect(DitherMatrix.fruit(sizeLog2: 4) == DitherMatrix.fruit(sizeLog2: 4))
        #expect(DitherMatrix.cachedFruit(sizeLog2: 4) == DitherMatrix.fruit(sizeLog2: 4))
    }

    @Test func fruitSizeIsClampedToWhatMpvAccepts() {
        #expect(DitherMatrix.fruit(sizeLog2: 0).count == 16)
    }

    /// Blue noise: neighbouring cells must not hold neighbouring ranks, which is what an ordered ramp
    /// (and a broken port) would give. The mean distance between a cell and its right neighbour is a third
    /// of the range for independent ranks.
    @Test func fruitNeighboursAreFarApartInRank() {
        let side = 64
        let matrix = DitherMatrix.fruit(sizeLog2: 6)
        var total: Float = 0
        for y in 0..<side {
            for x in 0..<side {
                total += abs(matrix[y * side + x] - matrix[y * side + (x + 1) % side])
            }
        }
        #expect(total / Float(side * side) > 0.3)
    }

    @Test func bayerIsTheClassicMatrix() {
        let four = DitherMatrix.bayer(size: 4).map { Int(($0 * 16).rounded()) }
        #expect(four == [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5])
        let sixteen = DitherMatrix.bayer(size: DitherMatrix.orderedSize)
        #expect(ranks(sixteen) == Array(0..<256))
    }
}

@Suite("Video dither options")
struct VideoDitherOptionsTests {

    @Test func defaultsAreMpvs() {
        let options = VideoDitherOptions()
        #expect(options.algorithm == .fruit)
        #expect(options.depth == .auto)
        #expect(options.fruitSizeLog2 == 6)
        #expect(!options.temporal)
        #expect(options.temporalPeriod == 1)
    }

    @Test func autoDepthIsTenBitsAndOthersSnapToEightOrTen() {
        #expect(VideoDitherOptions().resolvedBits == 10)
        #expect(VideoDitherOptions(depth: .bits(8)).resolvedBits == 8)
        #expect(VideoDitherOptions(depth: .bits(6)).resolvedBits == 8)
        #expect(VideoDitherOptions(depth: .bits(10)).resolvedBits == 10)
        #expect(VideoDitherOptions(depth: .bits(12)).resolvedBits == 10)
    }

    @Test func loadOptionIsOffByDefaultAndATuningField() {
        #expect(LoadOptions().videoDither == nil)
        #expect(SessionOptionCorrection.knownFields.contains("videoDither"))
        #expect(!SessionOptionCorrection.loadIdentityFields.contains("videoDither"))
    }

    @Test func debandDefaultsAreMpvs() {
        let options = VideoDebandOptions()
        #expect(options.iterations == 1)
        #expect(options.threshold == 48)
        #expect(options.range == 16)
        #expect(options.grain == 32)
    }

    @Test func debandLoadOptionIsOffByDefaultAndATuningField() {
        #expect(LoadOptions().videoDeband == nil)
        #expect(SessionOptionCorrection.knownFields.contains("videoDeband"))
        #expect(!SessionOptionCorrection.loadIdentityFields.contains("videoDeband"))
    }

    /// The Swift struct is copied into the shader's constant buffer byte for byte.
    @Test func shaderParamsLayoutMatchesTheShader() {
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.chromaOffset) == 32)
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.debandThreshold) == 40)
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.debandGrain) == 48)
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.matrixSize) == 56)
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.debandIterations) == 64)
        #expect(MemoryLayout<DitherShaderParams>.offset(of: \.debandRange) == 72)
        #expect(MemoryLayout<DitherShaderParams>.stride == 80)
    }

    /// mpv's threshold 48 is 48 / 16384 of the code range, its grain 32 is 32 / 8192; here they are in units
    /// of Y' (219 codes of 255 in video range) and chroma (224).
    @Test func debandParamsUseMpvsScales() {
        let p = VideoDitherer.params(layout: .init(bits: 10, videoRange: true), matrix: .bt709, bits: 10,
                                     matrixSize: 64, chromaOffset: .zero, phase: 0, deband: VideoDebandOptions(),
                                     debandRangeScale: 2, frameIndex: 7)
        #expect(abs(p.debandThreshold.x - 48 / 16384 * 1023 / 876) < 1e-7)
        #expect(abs(p.debandThreshold.y - 48 / 16384 * 1023 / 896) < 1e-7)
        #expect(abs(p.debandGrain.x - 32 / 8192 * 1023 / 876) < 1e-7)
        #expect(p.debandIterations == 1)
        #expect(p.debandRange == 32)
        #expect(p.frameIndex == 7)

        let off = VideoDitherer.params(layout: .init(bits: 10, videoRange: true), matrix: .bt709, bits: 10,
                                       matrixSize: 64, chromaOffset: .zero, phase: 0)
        #expect(off.debandIterations == 0 && off.debandGrain == .zero && off.debandThreshold == .zero)
        let shrunk = VideoDitherer.params(layout: .init(bits: 8, videoRange: false), matrix: .bt709, bits: 8,
                                          matrixSize: 16, chromaOffset: .zero, phase: 0,
                                          deband: VideoDebandOptions(iterations: 99, range: 16), debandRangeScale: 0.5)
        #expect(shrunk.debandIterations == 16)
        #expect(shrunk.debandRange == 16)
    }

    @Test func noAlgorithmMeansPlainRounding() {
        let p = VideoDitherer.params(layout: .init(bits: 10, videoRange: true), matrix: .bt709, bits: 10,
                                     matrixSize: 0, chromaOffset: .zero, phase: 0)
        #expect(p.matrixSize == 0)
        #expect(p.biasCentre == 0.5)
        #expect(VideoDitherer.phase(frame: 3, options: VideoDitherOptions(algorithm: .none, temporal: true)) == 0)
    }

    @Test func temporalPhaseCyclesThroughEightOrientations() {
        let off = VideoDitherOptions()
        #expect((0..<20).allSatisfy { VideoDitherer.phase(frame: $0, options: off) == 0 })
        let on = VideoDitherOptions(temporal: true)
        #expect((0..<10).map { VideoDitherer.phase(frame: $0, options: on) } == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        let slow = VideoDitherOptions(temporal: true, temporalPeriod: 3)
        #expect((0..<7).map { VideoDitherer.phase(frame: $0, options: slow) } == [0, 0, 0, 1, 1, 1, 2])
    }

    @Test func videoRangeTenBitCodesMapToZeroAndOne() {
        let p = VideoDitherer.params(layout: .init(bits: 10, videoRange: true), matrix: .bt709, bits: 10,
                                     matrixSize: 64, chromaOffset: .zero, phase: 0)
        func sample(_ code: Float) -> Float { code * 64 / 65535 }
        #expect(abs(sample(64) * p.yScale + p.yOffset) < 1e-5)
        #expect(abs(sample(940) * p.yScale + p.yOffset - 1) < 1e-5)
        #expect(abs(sample(512) * p.cScale + p.cOffset) < 1e-5)
        #expect(abs(sample(960) * p.cScale + p.cOffset - 0.5) < 1e-5)
        #expect(p.steps == 1023)
        #expect(p.biasCentre == 0.5 / 4096)
    }

    @Test func fullRangeEightBitCodesMapToZeroAndOne() {
        let p = VideoDitherer.params(layout: .init(bits: 8, videoRange: false), matrix: .bt601, bits: 8,
                                     matrixSize: 16, chromaOffset: .zero, phase: 0)
        #expect(abs(1 * p.yScale + p.yOffset - 1) < 1e-6)
        #expect(abs((128.0 / 255) * p.cScale + p.cOffset) < 1e-6)
        #expect(p.steps == 255)
        #expect(p.kr == 0.299)
    }

    @Test func frameHandlerDeliversTheOriginalWhenThePassDeclines() throws {
        var made: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &made)
        let original = try #require(made)
        var other: CVPixelBuffer?
        CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &other)
        let replacement = try #require(other)

        nonisolated(unsafe) var delivered: [CVPixelBuffer] = []
        let sink: DecodedFrameHandler = { buffer, _, _ in delivered.append(buffer) }
        DitheringFrameHandler.wrap(sink) { _ in nil }(original, .zero, nil)
        nonisolated(unsafe) let swapped = replacement
        DitheringFrameHandler.wrap(sink) { _ in swapped }(original, .zero, nil)
        #expect(delivered.count == 2)
        #expect(delivered[0] === original)
        #expect(delivered[1] === replacement)
    }
}

@Suite("Video dither pass", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct VideoDitherPassTests {

    // MARK: Buffers

    private func makeBuffer(_ format: OSType, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                                         kCVPixelBufferMetalCompatibilityKey as String: true]
        CVPixelBufferCreate(nil, width, height, format, attributes as CFDictionary, &buffer)
        return try #require(buffer)
    }

    /// A 10-bit 4:2:0 video-range frame; `code` gives (Y', Cb, Cr) as 10-bit codes for a luma position.
    private func p010(width: Int, height: Int, _ code: (Int, Int) -> (Int, Int, Int)) throws -> CVPixelBuffer {
        let buffer = try makeBuffer(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
        let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!
        let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        for y in 0..<height {
            let row = luma.advanced(by: y * lumaStride).assumingMemoryBound(to: UInt16.self)
            for x in 0..<width { row[x] = UInt16(code(x, y).0 << 6) }
        }
        for y in 0..<(height / 2) {
            let row = chroma.advanced(by: y * chromaStride).assumingMemoryBound(to: UInt16.self)
            for x in 0..<(width / 2) {
                let c = code(x * 2, y * 2)
                row[x * 2] = UInt16(c.1 << 6)
                row[x * 2 + 1] = UInt16(c.2 << 6)
            }
        }
        return buffer
    }

    /// An 8-bit 4:2:0 video-range frame of one colour.
    private func nv12(width: Int, height: Int, y: UInt8, cb: UInt8, cr: UInt8) throws -> CVPixelBuffer {
        let buffer = try makeBuffer(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        for row in 0..<height { for x in 0..<width { luma[row * lumaStride + x] = y } }
        for row in 0..<(height / 2) {
            for x in 0..<(width / 2) {
                chroma[row * chromaStride + x * 2] = cb
                chroma[row * chromaStride + x * 2 + 1] = cr
            }
        }
        return buffer
    }

    /// (R, G, B) codes of an output frame, 10-bit for `l10r` and 8-bit for BGRA, row-major.
    private func rgb(_ buffer: CVPixelBuffer) -> [(Int, Int, Int)] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let tenBit = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_ARGB2101010LEPacked
        var pixels: [(Int, Int, Int)] = []
        pixels.reserveCapacity(width * height)
        for y in 0..<height {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt32.self)
            for x in 0..<width {
                let v = row[x]
                if tenBit {
                    pixels.append((Int((v >> 20) & 0x3FF), Int((v >> 10) & 0x3FF), Int(v & 0x3FF)))
                } else {
                    pixels.append((Int((v >> 16) & 0xFF), Int((v >> 8) & 0xFF), Int(v & 0xFF)))
                }
            }
        }
        return pixels
    }

    private func mean(_ values: [Int]) -> Double { Double(values.reduce(0, +)) / Double(values.count) }

    // MARK: Depth

    /// Grey at 10-bit code 500 is 509.16 of 1023 and 126.92 of 255: between codes either way. Over whole
    /// tiles of the matrix the dithered mean lands on it; plain rounding would be off by up to half a code.
    @Test(arguments: [(VideoDitherOptions.Algorithm.fruit, 10), (.fruit, 8), (.ordered, 10), (.ordered, 8)])
    func flatGreyKeepsItsLevelOnAverage(_ algorithm: VideoDitherOptions.Algorithm, _ bits: Int) throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(algorithm: algorithm, depth: .bits(bits)), synchronousMatrix: true))
        let source = try p010(width: 128, height: 128) { _, _ in (500, 512, 512) }
        let output = try #require(ditherer.dither(source))
        #expect(CVPixelBufferGetPixelFormatType(output) == ditherer.outputPixelFormat)
        #expect(ditherer.outputBits == bits)

        let pixels = rgb(output)
        let reference = Double(500 - 64) / 876 * Double((1 << bits) - 1)
        for channel in [pixels.map(\.0), pixels.map(\.1), pixels.map(\.2)] {
            #expect(abs(mean(channel) - reference) < 0.01, "mean \(mean(channel)) vs \(reference)")
            #expect(Set(channel) == [Int(reference.rounded(.down)), Int(reference.rounded(.up))])
        }
        #expect(abs(reference.rounded() - reference) > 0.05, "the level must sit between codes for this to test anything")
    }

    @Test func tenBitOutputUsesCodesEightBitsCannotHold() throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(depth: .bits(10))))
        let source = try p010(width: 256, height: 64) { x, _ in (64 + x * 2, 512, 512) }
        let output = try #require(ditherer.dither(source))
        #expect(CVPixelBufferGetPixelFormatType(output) == kCVPixelFormatType_ARGB2101010LEPacked)
        let reds = Set(rgb(output).map(\.0))
        #expect(reds.contains { $0 % 4 != 0 })
        #expect(reds.count > 256)
    }

    @Test(arguments: [8, 10])
    func blackAndWhiteStayExact(_ bits: Int) throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(depth: .bits(bits))))
        let top = (1 << bits) - 1
        let black = rgb(try #require(ditherer.dither(try p010(width: 64, height: 64) { _, _ in (64, 512, 512) })))
        #expect(black.allSatisfy { $0 == (0, 0, 0) })
        let white = rgb(try #require(ditherer.dither(try p010(width: 64, height: 64) { _, _ in (940, 512, 512) })))
        #expect(white.allSatisfy { $0 == (top, top, top) })
    }

    // MARK: Colour

    /// BT.709 video-range red is Y' 63, Cb 102, Cr 240: the channels must come out in the right order.
    @Test(arguments: [8, 10])
    func eightBitRedDecodesToRed(_ bits: Int) throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(depth: .bits(bits))))
        let source = try nv12(width: 64, height: 64, y: 63, cb: 102, cr: 240)
        CVBufferSetAttachment(source, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        let pixels = rgb(try #require(ditherer.dither(source)))
        let top = Double((1 << bits) - 1)
        #expect(mean(pixels.map(\.0)) > top * 0.98)
        #expect(mean(pixels.map(\.1)) < top * 0.02)
        #expect(mean(pixels.map(\.2)) < top * 0.02)
    }

    /// The picture must look the same with the pass on as with it off, where the system converts the
    /// frame: a smooth colourful frame through VideoToolbox's own Y'CbCr to RGB conversion and through
    /// this pass differs by rounding and chroma filtering only.
    @Test func coloursMatchTheSystemsConversion() throws {
        let width = 256, height = 128
        let source = try p010(width: width, height: height) { x, y in
            (120 + x * 3, 400 + y, 620 - x / 2)
        }
        CVBufferSetAttachment(source, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(source, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(source, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)

        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &session)
        let transfer = try #require(session)
        let reference = try makeBuffer(kCVPixelFormatType_ARGB2101010LEPacked, width: width, height: height)
        try #require(VTPixelTransferSessionTransferImage(transfer, from: source, to: reference) == noErr)

        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(depth: .bits(10))))
        let ours = rgb(try #require(ditherer.dither(source)))
        let theirs = rgb(reference)
        var total = 0, worst = 0
        for (a, b) in zip(ours, theirs) {
            for d in [abs(a.0 - b.0), abs(a.1 - b.1), abs(a.2 - b.2)] {
                total += d
                worst = max(worst, d)
            }
        }
        let meanDifference = Double(total) / Double(ours.count * 3)
        #expect(meanDifference < 1.5, "mean difference \(meanDifference) of 1023")
        #expect(worst <= 8, "worst difference \(worst) of 1023")
    }

    @Test func temporalDitherMovesThePatternButNotTheLevel() throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(temporal: true)))
        let source = try p010(width: 64, height: 64) { _, _ in (500, 512, 512) }
        let first = rgb(try #require(ditherer.dither(source))).map(\.0)
        let second = rgb(try #require(ditherer.dither(source))).map(\.0)
        #expect(first != second)
        #expect(abs(mean(first) - mean(second)) < 0.001)

        let still = try #require(VideoDitherer(options: VideoDitherOptions()))
        let a = rgb(try #require(still.dither(source))).map(\.0)
        let b = rgb(try #require(still.dither(source))).map(\.0)
        #expect(a == b)
    }

    // MARK: Deband

    /// Per-column mean of the red codes.
    private func columnMeans(_ pixels: [(Int, Int, Int)], width: Int) -> [Double] {
        let rows = pixels.count / width
        return (0..<width).map { x in mean((0..<rows).map { pixels[$0 * width + x].0 }) }
    }

    /// Every channel of every pixel, in order, so two frames compare with `==`.
    private func codes(_ pixels: [(Int, Int, Int)]) -> [Int] { pixels.flatMap { [$0.0, $0.1, $0.2] } }

    private func largestStep(_ columns: [Double]) -> Double {
        zip(columns, columns.dropFirst()).map { abs($1 - $0) }.max() ?? 0
    }

    /// Bands two codes apart, 32 pixels wide, as an encoder leaves them in a gradient: within the
    /// threshold, so the deband pass spreads each step over its neighbourhood.
    @Test func debandSmoothsAStaircase() throws {
        let width = 256, height = 128
        let source = try p010(width: width, height: height) { x, _ in (300 + (x / 32) * 2, 512, 512) }
        let plain = try #require(VideoDitherer(options: VideoDitherOptions(), synchronousMatrix: true))
        let deband = try #require(VideoDitherer(options: VideoDitherOptions(), deband: VideoDebandOptions(grain: 0),
                                                synchronousMatrix: true))
        let before = largestStep(columnMeans(rgb(try #require(plain.dither(source))), width: width))
        let after = largestStep(columnMeans(rgb(try #require(deband.dither(source))), width: width))
        #expect(before > 2, "the bands must show without deband (\(before))")
        #expect(after < before * 0.5, "largest step between columns \(after) with deband, \(before) without")
    }

    /// An edge far above the threshold is real detail: it keeps its position and both levels.
    @Test func debandKeepsAHardEdge() throws {
        let width = 128, height = 128
        let source = try p010(width: width, height: height) { x, _ in (x < 64 ? 200 : 700, 512, 512) }
        let plain = try #require(VideoDitherer(options: VideoDitherOptions(), synchronousMatrix: true))
        let reference = rgb(try #require(plain.dither(source)))
        let clean = try #require(VideoDitherer(options: VideoDitherOptions(), deband: VideoDebandOptions(grain: 0),
                                               synchronousMatrix: true))
        #expect(codes(rgb(try #require(clean.dither(source)))) == codes(reference))

        let grained = try #require(VideoDitherer(options: VideoDitherOptions(), deband: VideoDebandOptions(),
                                                 synchronousMatrix: true))
        let columns = columnMeans(rgb(try #require(grained.dither(source))), width: width)
        let expected = columnMeans(reference, width: width)
        for (x, (got, want)) in zip(columns, expected).enumerated() {
            #expect(abs(got - want) < 1, "column \(x): \(got) vs \(want)")
        }
    }

    @Test func debandWithNothingToDoIsThePassWithoutIt() throws {
        let source = try p010(width: 128, height: 64) { x, y in (100 + x * 5, 400 + y, 620 - x) }
        let plain = try #require(VideoDitherer(options: VideoDitherOptions(), synchronousMatrix: true))
        let idle = try #require(VideoDitherer(options: VideoDitherOptions(),
                                              deband: VideoDebandOptions(iterations: 0, grain: 0), synchronousMatrix: true))
        #expect(codes(rgb(try #require(idle.dither(source)))) == codes(rgb(try #require(plain.dither(source)))))
    }

    @Test func grainLeavesBlackAndNeutralGreyAlone() throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(), deband: VideoDebandOptions(grain: 128),
                                                  synchronousMatrix: true))
        let black = rgb(try #require(ditherer.dither(try p010(width: 64, height: 64) { _, _ in (64, 512, 512) })))
        #expect(black.allSatisfy { $0 == (0, 0, 0) })

        let grey = rgb(try #require(ditherer.dither(try p010(width: 128, height: 128) { _, _ in (500, 512, 512) })))
        let reference = Double(500 - 64) / 876 * 1023
        let channels = [grey.map(\.0), grey.map(\.1), grey.map(\.2)]
        #expect(Set(channels[0]).count > 4, "grain must show")
        for channel in channels {
            #expect(abs(mean(channel) - reference) < 0.2, "mean \(mean(channel)) vs \(reference)")
        }
        #expect(abs(mean(channels[0]) - mean(channels[1])) < 0.02)
        #expect(abs(mean(channels[2]) - mean(channels[1])) < 0.02)
    }

    /// The grain is seeded with the frame index, so it does not stand still as a fixed pattern, also
    /// without temporal dithering.
    @Test func grainChangesFromFrameToFrame() throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(), deband: VideoDebandOptions(),
                                                  synchronousMatrix: true))
        let source = try p010(width: 64, height: 64) { _, _ in (500, 512, 512) }
        let first = rgb(try #require(ditherer.dither(source))).map(\.0)
        let second = rgb(try #require(ditherer.dither(source))).map(\.0)
        #expect(first != second)
    }

    @Test(arguments: [8, 10])
    func noAlgorithmRoundsToNearest(_ bits: Int) throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(algorithm: .none, depth: .bits(bits)),
                                                  synchronousMatrix: true))
        let source = try p010(width: 64, height: 64) { _, _ in (500, 512, 512) }
        let reference = Double(500 - 64) / 876 * Double((1 << bits) - 1)
        let pixels = rgb(try #require(ditherer.dither(source)))
        let expected = Int(reference.rounded())
        #expect(pixels.allSatisfy { $0 == (expected, expected, expected) })
    }

    /// Debanding alone, as the software host builds it: no dither noise, but the bands still go.
    @Test func debandWithoutDitherStillSmooths() throws {
        let width = 256, height = 128
        let source = try p010(width: width, height: height) { x, _ in (300 + (x / 32) * 2, 512, 512) }
        let plain = try #require(VideoDitherer(options: VideoDitherOptions(algorithm: .none), synchronousMatrix: true))
        let deband = try #require(VideoDitherer(options: VideoDitherOptions(algorithm: .none), deband: VideoDebandOptions(),
                                                synchronousMatrix: true))
        let before = largestStep(columnMeans(rgb(try #require(plain.dither(source))), width: width))
        let after = largestStep(columnMeans(rgb(try #require(deband.dither(source))), width: width))
        #expect(after < before * 0.5, "largest step between columns \(after) with deband, \(before) without")
    }

    // MARK: What it leaves alone

    @Test func hdrAndUnknownFormatsAreNotTaken() throws {
        let ditherer = try #require(VideoDitherer())
        let pq = try p010(width: 16, height: 16) { _, _ in (500, 512, 512) }
        CVBufferSetAttachment(pq, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ, .shouldPropagate)
        #expect(ditherer.dither(pq) == nil)
        let hlg = try p010(width: 16, height: 16) { _, _ in (500, 512, 512) }
        CVBufferSetAttachment(hlg, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_2100_HLG, .shouldPropagate)
        #expect(ditherer.dither(hlg) == nil)
        #expect(ditherer.dither(try makeBuffer(kCVPixelFormatType_32BGRA, width: 16, height: 16)) == nil)
    }

    @Test func outputCarriesColourTagsAndDropsTheMatrix() throws {
        let ditherer = try #require(VideoDitherer())
        let source = try p010(width: 16, height: 16) { _, _ in (500, 512, 512) }
        CVBufferSetAttachment(source, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)
        CVBufferSetAttachment(source, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_SMPTE_C, .shouldPropagate)
        CVBufferSetAttachment(source, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        let par = [kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String: 8,
                   kCVImageBufferPixelAspectRatioVerticalSpacingKey as String: 9]
        CVBufferSetAttachment(source, kCVImageBufferPixelAspectRatioKey, par as CFDictionary, .shouldPropagate)
        let output = try #require(ditherer.dither(source))
        #expect(CVBufferCopyAttachment(output, kCVImageBufferYCbCrMatrixKey, nil) == nil)
        #expect(CVBufferCopyAttachment(output, kCVImageBufferColorPrimariesKey, nil) as? String
                == kCVImageBufferColorPrimaries_SMPTE_C as String)
        #expect(CVBufferCopyAttachment(output, kCVImageBufferTransferFunctionKey, nil) as? String
                == kCVImageBufferTransferFunction_ITU_R_709_2 as String)
        #expect((CVBufferCopyAttachment(output, kCVImageBufferPixelAspectRatioKey, nil) as? [String: Int]) == par)
    }

    /// A fruit matrix the process has not built must not hold the first frame back: the pass starts on the
    /// ordered matrix and still dithers.
    @Test func aPassWithoutItsFruitMatrixStillDithers() throws {
        let ditherer = try #require(VideoDitherer(options: VideoDitherOptions(depth: .bits(8), fruitSizeLog2: 5)))
        let source = try p010(width: 64, height: 64) { _, _ in (500, 512, 512) }
        let reds = rgb(try #require(ditherer.dither(source))).map(\.0)
        #expect(Set(reds) == [126, 127])
        #expect(abs(mean(reds) - 126.917) < 0.01)
    }

    @Test func untaggedInputIsTaggedBT709() throws {
        let ditherer = try #require(VideoDitherer())
        let output = try #require(ditherer.dither(try p010(width: 16, height: 16) { _, _ in (500, 512, 512) }))
        #expect(CVBufferCopyAttachment(output, kCVImageBufferColorPrimariesKey, nil) as? String
                == kCVImageBufferColorPrimaries_ITU_R_709_2 as String)
        #expect(CVBufferCopyAttachment(output, kCVImageBufferTransferFunctionKey, nil) as? String
                == kCVImageBufferTransferFunction_ITU_R_709_2 as String)
    }
}
