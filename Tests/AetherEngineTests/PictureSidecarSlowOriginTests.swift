import Testing
import Foundation
@testable import AetherEngine

/// A picture-based sidecar over HTTP whose origin produces the file on request (Jellyfin extracts every
/// subtitle stream of an item before it answers the first request for an embedded PGS or VobSub track)
/// is waited for on one ranged request before the reader opens on it.
///
/// Before: the reader gave the origin seconds for its first byte. A film-length extraction takes
/// minutes, so the request timed out with nothing read, the retries were abandoned the same way, and
/// the selected track showed nothing.
@Suite("Picture sidecar from a slow origin", .timeLimit(.minutes(1)))
struct PictureSidecarSlowOriginTests {

    @Test("The origin is waited for on one request asking for one byte")
    func waitIsOneRangedRequest() async throws {
        let origin = try #require(RangedOrigin(files: ["/Subtitles/4/Stream.pgssub": Data([1, 2, 3])],
                                               delaySeconds: 1.5))
        defer { origin.stop() }
        try await SidecarOriginWait.wait(for: origin.url("/Subtitles/4/Stream.pgssub"), headers: ["X-Token": "t"])
        #expect(origin.requests == [.init(path: "/Subtitles/4/Stream.pgssub", range: "bytes=0-0", token: "t")])
    }

    @Test("A refused sidecar fails typed with its status")
    func refusalIsTyped() async throws {
        let origin = try #require(RangedOrigin(files: [:]))
        defer { origin.stop() }
        await #expect(throws: SubtitleDecoderError.self) {
            try await SidecarOriginWait.wait(for: origin.url("/missing.sup"), headers: [:])
        }
    }

    @Test("Cancelling the track ends the wait")
    func cancellationEndsTheWait() async throws {
        let origin = try #require(RangedOrigin(files: ["/s.sup": Data([1])], delaySeconds: 30))
        defer { origin.stop() }
        let wait = Task { try await SidecarOriginWait.wait(for: origin.url("/s.sup"), headers: [:]) }
        try await waitFor { origin.requests.count == 1 }
        wait.cancel()
        await #expect(throws: (any Error).self) { try await wait.value }
    }

    @Test("A .sup behind a slow origin is read for its packets once the origin answers")
    func slowSupDecodes() async throws {
        let sup = Issue921SidecarBitmapDrainTests.makeSup(sets: 5)
        let origin = try #require(RangedOrigin(files: ["/film/4/Stream.pgssub": sup], delaySeconds: 1.5))
        defer { origin.stop() }
        let result = try await SubtitleDecoder.decodeFile(
            url: origin.url("/film/4/Stream.pgssub"), keepsBitmapPackets: true, formatHint: "sup")
        let source = try #require(result.bitmapSource)
        #expect(source.packets.count == 10)
        #expect(origin.requests.first?.range == "bytes=0-0")
    }

    @Test("Text sidecars keep their own whole-body fetch, local files are not probed")
    func onlyRemotePictureSidecarsWait() {
        #expect(SidecarOriginWait.appliesTo(URL(string: "http://jf/Stream.pgssub")!, formatHint: "sup"))
        #expect(SidecarOriginWait.appliesTo(URL(string: "https://jf/Stream.dvdsub")!, formatHint: nil))
        #expect(!SidecarOriginWait.appliesTo(URL(string: "http://jf/Stream.ass")!, formatHint: nil))
        #expect(!SidecarOriginWait.appliesTo(URL(string: "http://jf/streams/7")!, formatHint: "srt"))
        #expect(!SidecarOriginWait.appliesTo(URL(fileURLWithPath: "/tmp/a.sup"), formatHint: nil))
    }

    @Test("The budget outlasts one extraction pass over a film-length remux")
    func budgetIsMinutes() {
        #expect(SidecarOriginWait.firstResponseTimeout >= 300)
        let configuration = SidecarOriginWait.sessionConfiguration()
        #expect(configuration.timeoutIntervalForRequest == SidecarOriginWait.firstResponseTimeout)
        #expect(configuration.timeoutIntervalForResource == SidecarOriginWait.firstResponseTimeout)
    }
}

/// A loopback origin that serves fixed binary bodies by path, honouring single `bytes=a-b` / `bytes=a-`
/// ranges with 206, after an optional delay per request. Records each request's path, Range and
/// `X-Token` header. A path it does not know answers 404.
private final class RangedOrigin: @unchecked Sendable {
    struct Request: Equatable {
        let path: String
        let range: String?
        var token: String? = nil
    }

    private let listener: LoopbackListener
    private let files: [String: Data]
    private let delayMicros: useconds_t
    private let lock = NSLock()
    private var log: [Request] = []

    init?(files: [String: Data], delaySeconds: Double = 0) {
        guard let listener = LoopbackListener() else { return nil }
        self.listener = listener
        self.files = files
        delayMicros = useconds_t(delaySeconds * 1_000_000)
        listener.start { [self] fd in
            Thread.detachNewThread { [self] in serve(fd) }
            return true
        }
    }

    var requests: [Request] { lock.withLock { log } }

    func url(_ path: String) -> URL { URL(string: "http://127.0.0.1:\(listener.port)\(path)")! }

    func stop() { listener.stop() }

    private func serve(_ fd: Int32) {
        defer { close(fd) }
        var head = Data()
        var byte: UInt8 = 0
        while !head.suffix(4).elementsEqual([13, 10, 13, 10]) {
            guard read(fd, &byte, 1) == 1 else { return }
            head.append(byte)
        }
        let lines = String(decoding: head, as: UTF8.self).components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        guard parts.count >= 2 else { return }
        let path = String(parts[1])
        func header(_ name: String) -> String? {
            lines.dropFirst().first { $0.lowercased().hasPrefix(name + ":") }
                .map { String($0.dropFirst(name.count + 1)).trimmingCharacters(in: .whitespaces) }
        }
        let range = header("range")
        lock.withLock { log.append(Request(path: path, range: range, token: header("x-token"))) }
        if delayMicros > 0 { usleep(delayMicros) }
        var response: Data
        if let body = files[path] {
            if let range, let span = Self.span(range, count: body.count) {
                let slice = body[span]
                response = Data(("HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\n"
                    + "Content-Range: bytes \(span.lowerBound)-\(span.upperBound - 1)/\(body.count)\r\n"
                    + "Content-Length: \(slice.count)\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n").utf8)
                response.append(slice)
            } else {
                response = Data(("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                    + "Content-Length: \(body.count)\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n").utf8)
                response.append(body)
            }
        } else {
            response = Data("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".utf8)
        }
        var sent = 0
        while sent < response.count {
            let n = response[sent...].withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
            guard n > 0 else { return }
            sent += n
        }
    }

    /// `bytes=a-b` or `bytes=a-` as a half-open span of the body, nil when it does not fit.
    private static func span(_ range: String, count: Int) -> Range<Int>? {
        guard range.hasPrefix("bytes=") else { return nil }
        let bounds = range.dropFirst("bytes=".count).split(separator: "-", omittingEmptySubsequences: false)
        guard bounds.count == 2, let start = Int(bounds[0]), start < count else { return nil }
        let end = Int(bounds[1]).map { min($0, count - 1) } ?? count - 1
        return end >= start ? start..<(end + 1) : nil
    }
}
