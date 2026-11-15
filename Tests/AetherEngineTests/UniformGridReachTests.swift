import Testing
import AetherLibavutil
@testable import AetherEngine

/// A uniform grid measured off an index-less MPEG-TS follows the source's keyframes only as closely
/// as one spacing measurement allows. On a 23.976 fps progressive transcode with a 250-frame GOP the
/// stride came out half a tick long, which put boundary 134 sixty-six ticks after the keyframe it
/// stands for: the cutter doubled a segment, and a restart there opened a whole GOP late, so a seek
/// into the last GOP left AVPlayer waiting on content that never came and the item never ended.
@Suite("A measured uniform grid reaches the keyframe sitting just below its boundary")
struct UniformGridReachTests {
    private let tb = AVRational(num: 1, den: 90_000)

    @Test("only a measured grid has a reach, held to a quarter of the stride")
    func reachOnlyForMeasuredGrids() {
        #expect(HLSVideoEngine.uniformGridReachTicks(spacing: .measured(10.427), strideSeconds: 10.427,
                                                     videoTimeBase: tb) == 45_000)
        #expect(HLSVideoEngine.uniformGridReachTicks(spacing: .measured(1.0), strideSeconds: 1.0,
                                                     videoTimeBase: tb) == 22_500)
        #expect(HLSVideoEngine.uniformGridReachTicks(spacing: .unknown, strideSeconds: 4,
                                                     videoTimeBase: tb) == 0)
        #expect(HLSVideoEngine.uniformGridReachTicks(spacing: .exceedsBudget(30), strideSeconds: 30,
                                                     videoTimeBase: tb) == 0)
        #expect(HLSVideoEngine.uniformGridReachTicks(spacing: .measured(10), strideSeconds: 10,
                                                     videoTimeBase: AVRational(num: 0, den: 1)) == 0)
    }

    @Test("the reporting file: boundary 134 sits 66 ticks past its keyframe")
    func keyframeJustBelowItsBoundaryOpensItsSegment() {
        // Anchor 129840; the grid strides 938438 ticks, the keyframes come every 938437.5.
        let anchor: Int64 = 129_840
        let boundaries = (0...136).map { anchor + Int64($0) * 938_438 }
        let keyframe134: Int64 = 125_880_465
        #expect(boundaries[134] - keyframe134 == 67)

        let reach = HLSVideoEngine.uniformGridReachTicks(spacing: .measured(10.427), strideSeconds: 10.427,
                                                         videoTimeBase: tb)
        // The restart gate takes it with the reach and drops it without.
        #expect(!HLSSegmentProducer.videoGateTargetSatisfied(pts: keyframe134, dts: keyframe134,
                                                             targetPts: boundaries[134]))
        #expect(HLSSegmentProducer.videoGateTargetSatisfied(pts: keyframe134, dts: keyframe134,
                                                            targetPts: boundaries[134], reachTicks: reach))
        // The keyframe a whole GOP earlier never reaches.
        #expect(!HLSSegmentProducer.videoGateTargetSatisfied(pts: keyframe134 - 938_437, dts: keyframe134 - 938_437,
                                                             targetPts: boundaries[134], reachTicks: reach))

        // Sequential cutting: without a reach the segment before swallows the GOP, with it every
        // keyframe opens the segment its boundary names.
        var without = VODSegmentCutter(sourceBoundaries: Array(boundaries[133...136]), planAnchorPts: anchor,
                                       baseIndex: 133)
        #expect(without.index(pts: keyframe134 - anchor, isKeyframe: true) == 133)
        var with = VODSegmentCutter(sourceBoundaries: Array(boundaries[133...136]), planAnchorPts: anchor,
                                    baseIndex: 133, reachTicks: Array(repeating: reach, count: 4))
        #expect(with.index(pts: keyframe134 - anchor, isKeyframe: true) == 134)
        #expect(with.index(pts: keyframe134 - anchor + 3754, isKeyframe: false) == 134)
    }

    @Test("a restart seeks below the boundary by the reach, never below the plan's start")
    func restartSeeksBelowTheBoundary() {
        #expect(HLSVideoEngine.restartSeekPts(boundaryPts: 125_880_531, uniformGridReachTicks: 45_000,
                                              planStartPts: 129_840) == 125_835_531)
        #expect(HLSVideoEngine.restartSeekPts(boundaryPts: 125_880_531, uniformGridReachTicks: 0,
                                              planStartPts: 129_840) == 125_880_531)
        #expect(HLSVideoEngine.restartSeekPts(boundaryPts: 140_000, uniformGridReachTicks: 45_000,
                                              planStartPts: 129_840) == 129_840)
        #expect(HLSVideoEngine.restartSeekPts(boundaryPts: 129_840, uniformGridReachTicks: 45_000,
                                              planStartPts: 129_840) == 129_840)
        #expect(HLSVideoEngine.restartSeekPts(boundaryPts: Int64.min + 5, uniformGridReachTicks: 45_000,
                                              planStartPts: Int64.min + 5) == Int64.min + 5)
    }
}
