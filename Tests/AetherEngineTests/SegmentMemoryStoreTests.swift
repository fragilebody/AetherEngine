import Foundation
import Testing
import AetherLibavformat
import AetherLibavcodec
import AetherLibavutil
@testable import AetherEngine

/// Fork (memory-segments): `LoadOptions.segmentMemoryBudgetBytes` keeps loopback segments in purgeable
/// memory instead of writing each one to disk, spills to disk above the budget, and treats a buffer the
/// system purged like a segment file that vanished (AE#451), so the provider produces it again.
@Suite("Segment memory store", .serialized)
struct SegmentMemoryStoreTests {

    private func makeCache(memoryBudget: Int?, forwardWindow: Int = 5, backwardWindow: Int = 5,
                           retentionBudget: Int = 0) -> SegmentCache {
        SegmentCache(forwardWindow: forwardWindow, backwardWindow: backwardWindow,
                     retentionBudgetBytes: retentionBudget,
                     baseDirectory: FileManager.default.temporaryDirectory
                        .appendingPathComponent("memory-store-\(UUID().uuidString)", isDirectory: true),
                     memoryBudgetBytes: memoryBudget)
    }

    private func bytes(_ count: Int, seed: UInt8) -> Data {
        Data((0..<count).map { UInt8(truncatingIfNeeded: Int(seed) &+ $0) })
    }

    private func append(_ data: Data, to buffer: MemorySegmentBuffer) -> Bool {
        data.withUnsafeBytes { buffer.append($0.baseAddress!, count: data.count) }
    }

    /// What the producer does for one segment the cache admits into memory.
    @discardableResult
    private func produceInMemory(_ cache: SegmentCache, index: Int, _ data: Data) throws -> MemorySegmentBuffer {
        let buffer = try #require(cache.admitMemorySegment(index: index))
        #expect(append(data, to: buffer))
        #expect(cache.adopt(index: index, memory: buffer, byteCount: data.count))
        return buffer
    }

    /// What the producer does for a segment admission sent to disk.
    private func produceOnDisk(_ cache: SegmentCache, index: Int, _ data: Data) throws {
        let staging = cache.sessionDir.appendingPathComponent("staging-seg-\(index)-\(UUID().uuidString.prefix(8)).tmp")
        try data.write(to: staging)
        #expect(cache.adopt(index: index, stagingPath: staging, byteCount: data.count))
    }

    private func segmentFiles(_ cache: SegmentCache) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: cache.sessionDir.path)) ?? [])
            .filter { $0.hasSuffix(".m4s") || $0.hasSuffix(".tmp") }
    }

    // MARK: - Store, serve, evict

    @Test("a segment admitted into memory is served from memory and writes no file")
    func memoryStoreAndServe() throws {
        let cache = makeCache(memoryBudget: 10_000)
        defer { cache.close() }
        let payload = bytes(1000, seed: 3)
        try produceInMemory(cache, index: 0, payload)

        #expect(cache.hasSegment(index: 0))
        #expect(cache.isMemoryResident(index: 0))
        #expect(cache.peekURL(index: 0) == nil, "a memory segment has no file for the sendfile path")
        #expect(cache.peek(index: 0) == payload)
        #expect(cache.fetch(index: 0, timeout: 0.1) == payload)
        guard case .data(let served) = cache.fetchSource(index: 0, timeout: 0.1, progressive: false) else {
            Issue.record("a stored memory segment is served whole"); return
        }
        #expect(served == payload)
        #expect(cache.totalBytes == 0, "the disk ledger bills nothing for a memory segment")
        #expect(cache.memoryEntryBytes == payload.count)
        #expect(cache.memoryBytes == payload.count)
        #expect(cache.diskBytes() == 0)
        #expect(segmentFiles(cache).isEmpty)
    }

    @Test("evicting a memory segment frees its bytes once the last serve lets go")
    func memoryEviction() throws {
        let cache = makeCache(memoryBudget: 10_000, forwardWindow: 2, backwardWindow: 2)
        defer { cache.close() }
        for i in 0..<3 { try produceInMemory(cache, index: i, bytes(500, seed: UInt8(i))) }
        #expect(cache.memoryBytes == 1500)

        var held = cache.fetch(index: 0, timeout: 0.1)
        cache.declareTarget(10)    // window-only pruning: 0...2 fall outside [8, 12]
        #expect(!cache.hasSegment(index: 0))
        #expect(cache.count == 0)
        #expect(cache.memoryEntryBytes == 0)
        #expect(cache.memoryBytes == 500, "the bytes a serve still holds stay resident and counted")
        #expect(held?.count == 500)
        held = nil
        #expect(cache.memoryBytes == 0)
    }

    @Test("watched history in memory is evicted farthest from the target first, within the memory budget")
    func memoryHistoryEvictedFarthestFirst() throws {
        // Retention on, hard window [target - 1, target + 1]: everything else is history.
        let cache = makeCache(memoryBudget: 2500, forwardWindow: 1, backwardWindow: 1, retentionBudget: 1 << 30)
        defer { cache.close() }
        for i in 0..<5 {
            cache.declareTarget(i)
            try produceInMemory(cache, index: i, bytes(500, seed: UInt8(i)))
        }
        cache.declareTarget(5)
        // 2500 bytes fit: all five stay.
        #expect((0..<5).allSatisfy { cache.hasSegment(index: $0) })
        try produceInMemory(cache, index: 5, bytes(500, seed: 5))
        // 3000 > 2500 once 5 lands: the farthest history (0) went to make room.
        #expect(!cache.hasSegment(index: 0))
        #expect((1...5).allSatisfy { cache.hasSegment(index: $0) })
        #expect(cache.memoryBytes <= 2500)
    }

    // MARK: - Spill

    @Test("a segment that would exceed the budget spills to disk and later ones return to memory")
    func spillAtTheBudget() throws {
        let cache = makeCache(memoryBudget: 250, forwardWindow: 10, backwardWindow: 2)
        defer { cache.close() }
        try produceInMemory(cache, index: 0, bytes(100, seed: 0))
        try produceInMemory(cache, index: 1, bytes(100, seed: 1))
        // 200 resident + ~100 for the next one > 250, and everything resident is in the hard window.
        #expect(cache.admitMemorySegment(index: 2) == nil)
        #expect(cache.memorySpillCount == 1)
        try produceOnDisk(cache, index: 2, bytes(100, seed: 2))
        #expect(!cache.isMemoryResident(index: 2))
        #expect(cache.totalBytes == 100)
        #expect(cache.peekURL(index: 2) != nil, "a spilled segment keeps the file path")
        #expect(cache.peek(index: 2) == bytes(100, seed: 2))

        // The consumer moves on and window pruning frees the memory: the next segment is in memory again.
        cache.declareTarget(8)
        #expect(cache.memoryBytes == 0)
        try produceInMemory(cache, index: 8, bytes(100, seed: 8))
        #expect(cache.isMemoryResident(index: 8))
    }

    @Test("a zero budget sends every segment to disk")
    func zeroBudgetSpillsEverything() {
        let cache = makeCache(memoryBudget: 0)
        defer { cache.close() }
        #expect(cache.storesSegmentsInMemory)
        #expect(cache.admitMemorySegment(index: 0) == nil)
    }

    // MARK: - Purgeable

    @Test("only watched history is purgeable, and a serve pins it while it reads")
    func purgeabilityFollowsTheTarget() throws {
        let cache = makeCache(memoryBudget: 10_000, forwardWindow: 5, backwardWindow: 5)
        defer { cache.close() }
        cache.declareTarget(3)
        let ahead = try produceInMemory(cache, index: 4, bytes(300, seed: 4))
        #expect(!ahead.isMarkedPurgeable, "not fetched yet: pinned")
        cache.declareTarget(5)
        #expect(ahead.isMarkedPurgeable, "behind the target: watched history")

        var served = cache.fetch(index: 4, timeout: 0.1)
        #expect(!ahead.isMarkedPurgeable, "a serve pins the buffer")
        #expect(served == bytes(300, seed: 4))
        served = nil
        #expect(ahead.isMarkedPurgeable, "released again once the serve lets go")

        cache.declareTarget(2)   // a backward seek puts it ahead of the consumer again
        #expect(!ahead.isMarkedPurgeable)
    }

    @Test("a purged buffer reads as a missing segment and is produced again")
    func purgedBufferIsReproduced() throws {
        let cache = makeCache(memoryBudget: 10_000)
        defer { cache.close() }
        let first = try produceInMemory(cache, index: 0, bytes(400, seed: 1))
        cache.declareTarget(2)
        first.purgeForTesting()

        #expect(cache.fetch(index: 0, timeout: 0) == nil, "a purged buffer is never served as zeros")
        #expect(!cache.hasSegment(index: 0), "the entry is dropped, so the provider re-produces it")
        #expect(cache.memoryPurgeCount == 1)
        #expect(cache.memoryEntryBytes == 0)
        #expect(cache.highestStoredIndex == 0, "how far the producer got is unchanged")

        try produceInMemory(cache, index: 0, bytes(400, seed: 9))
        #expect(cache.fetch(index: 0, timeout: 0.1) == bytes(400, seed: 9))
    }

    // MARK: - Progressive delivery

    @Test("a progressive reader follows a growing memory buffer and finishes at the seal")
    func progressiveReadOfGrowingBuffer() throws {
        let cache = makeCache(memoryBudget: 10_000)
        defer { cache.close() }
        let buffer = try #require(cache.admitMemorySegment(index: 3))
        let first = bytes(1000, seed: 1), second = bytes(700, seed: 9)
        #expect(append(first, to: buffer))
        cache.beginInProgress(index: 3, staging: .memory(buffer))

        guard case .progressive(let reader) = cache.fetchSource(index: 3, timeout: 1, progressive: true) else {
            Issue.record("a memory segment being written is served by a reader"); return
        }
        #expect(reader.next() == .bytes(first))
        #expect(append(second, to: buffer))
        #expect(reader.next() == .bytes(second))
        cache.adopt(index: 3, memory: buffer, byteCount: first.count + second.count)
        #expect(reader.next() == .finished)
        guard case .data(let sealed) = cache.fetchSource(index: 3, timeout: 1, progressive: true) else {
            Issue.record("an adopted memory segment is served whole"); return
        }
        #expect(sealed == first + second)
        #expect(segmentFiles(cache).isEmpty)
    }

    @Test("a buffer that grows past its first region keeps every byte")
    func bufferGrowth() throws {
        let buffer = try #require(MemorySegmentBuffer(index: 0, initialCapacity: 0))
        var expected = Data()
        // Past the 1 MiB minimum region twice over, in muxer-sized writes.
        for i in 0..<40 {
            let chunk = bytes(65_536, seed: UInt8(i))
            expected.append(chunk)
            #expect(append(chunk, to: buffer))
        }
        #expect(buffer.length == expected.count)
        #expect(buffer.pinnedData() == expected)
    }

    // MARK: - nil budget

    @Test("without a memory budget the cache keeps the disk path")
    func nilBudgetKeepsDisk() throws {
        let cache = makeCache(memoryBudget: nil)
        defer { cache.close() }
        #expect(!cache.storesSegmentsInMemory)
        #expect(cache.admitMemorySegment(index: 0) == nil)
        try produceOnDisk(cache, index: 0, bytes(100, seed: 0))
        #expect(cache.peekURL(index: 0) != nil)
        #expect(cache.totalBytes == 100)
        #expect(cache.memoryBytes == 0)
    }

    // MARK: - Muxer

    @Test("the muxer writes the same segment bytes into memory as into a staging file")
    func muxerMemoryStagingMatchesFile() throws {
        let fileRun = try Self.muxTwoSegments(inMemory: false)
        let memoryRun = try Self.muxTwoSegments(inMemory: true)
        guard case .completed(.file(let path), let fileBytes) = fileRun.outcome else {
            Issue.record("the file run cuts into a staging file"); return
        }
        guard case .completed(.memory(let buffer), let memoryBytes) = memoryRun.outcome else {
            Issue.record("the memory run cuts into a memory buffer"); return
        }
        #expect(fileBytes == memoryBytes)
        #expect(try Data(contentsOf: path) == buffer.pinnedData())
        #expect(memoryRun.filesLeft.isEmpty, "a memory-staged muxer opens no staging file")
        #expect(fileRun.initBytes == memoryRun.initBytes)
    }

    private struct MuxRun {
        let outcome: MP4SegmentMuxer.CutOutcome
        let initBytes: Data
        let filesLeft: [String]
    }

    private static func muxTwoSegments(inMemory: Bool) throws -> MuxRun {
        let demuxer = Demuxer()
        defer { demuxer.close() }
        try demuxer.open(
            reader: DataIOReader(data: Data(
                base64Encoded: AtmosDetectionProbeIntegrationTests.videoOnlyBase64,
                options: .ignoreUnknownCharacters) ?? Data()),
            formatHint: "mp4")
        let vStream = try #require(demuxer.stream(at: demuxer.videoStreamIndex))
        var packets: [UnsafeMutablePointer<AVPacket>] = []
        defer { for p in packets { var o: UnsafeMutablePointer<AVPacket>? = p; trackedPacketFree(&o) } }
        while let pkt = try demuxer.readPacket() {
            if pkt.pointee.stream_index == demuxer.videoStreamIndex, packets.count < 3 {
                packets.append(pkt)
            } else {
                var o: UnsafeMutablePointer<AVPacket>? = pkt
                trackedPacketFree(&o)
            }
        }
        try #require(packets.count == 3)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-muxer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        var initBytes = Data()
        let muxer = try MP4SegmentMuxer(
            initialSegmentIndex: 0, sessionDir: dir,
            video: MP4SegmentMuxer.VideoConfig(codecpar: UnsafePointer(vStream.pointee.codecpar),
                                               timeBase: vStream.pointee.time_base, codecTagOverride: nil),
            audio: nil,
            memoryStaging: inMemory ? { MemorySegmentBuffer(index: $0) } : nil,
            onInitCaptured: { initBytes = $0 })
        let frame = av_rescale_q(1, AVRational(num: 1, den: 25), muxer.muxerVideoTimeBase)
        for (i, source) in packets.enumerated() {
            guard let pkt = av_packet_clone(source) else { continue }
            pkt.pointee.stream_index = muxer.videoOutputStreamIndex
            pkt.pointee.dts = Int64(i) * frame
            pkt.pointee.pts = pkt.pointee.dts
            pkt.pointee.duration = frame
            if i == 0 { pkt.pointee.flags |= AV_PKT_FLAG_KEY }
            _ = muxer.writePacket(pkt)
            var o: UnsafeMutablePointer<AVPacket>? = pkt
            av_packet_free(&o)
        }
        let outcome = muxer.cutFragmentForNextSegment(1)
        // Read the file run's segment before the directory goes.
        if case .completed(.file(let path), _) = outcome {
            let copy = dir.deletingLastPathComponent().appendingPathComponent("memory-muxer-\(UUID().uuidString).m4s")
            try FileManager.default.copyItem(at: path, to: copy)
            let filesLeft = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            _ = muxer.finalize()
            return MuxRun(outcome: .completed(staging: .file(copy), bytesWritten: try Self.bytesWritten(outcome)),
                          initBytes: initBytes, filesLeft: filesLeft)
        }
        let filesLeft = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        _ = muxer.finalize()
        return MuxRun(outcome: outcome, initBytes: initBytes, filesLeft: filesLeft)
    }

    private static func bytesWritten(_ outcome: MP4SegmentMuxer.CutOutcome) throws -> Int {
        guard case .completed(_, let n) = outcome else { throw CancellationError() }
        return n
    }
}
