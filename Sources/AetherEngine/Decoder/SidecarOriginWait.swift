import Foundation

/// A picture-based sidecar over HTTP (a PGS `.sup`, a VobSub `.mks`, any subtitle container) is read
/// by the playback reader, which ranges through it rather than holding it whole (#921, #266). That
/// reader gives an origin seconds for its first byte, which is right for media and wrong for a media
/// server that produces the file on request: Jellyfin answers the first request for an embedded PGS
/// or VobSub track only once it has extracted every subtitle stream of the item, in one pass over the
/// whole file. On a Blu-ray remux that pass takes minutes, the reader gave up long before, and the
/// track showed nothing (measured on a 38 Mbit/s film: no byte within the reader's budget, and the
/// retries were abandoned the same way).
///
/// So before the reader opens, the origin is asked for one byte, and that one request is waited for
/// patiently (the text path, AE#984, waits the same way for the same reason). Once the origin has
/// answered, the file exists and the reader opens on it as before. The probe ends as soon as the
/// response head arrives: no body is read, whether or not the origin honours the range.
enum SidecarOriginWait {

    /// No response for this long ends the wait. A sidecar is optional and nothing waits on it but its
    /// own track, so the budget is set by the slowest extraction worth waiting for: one pass over a
    /// film-length remux on a server's disk.
    static let firstResponseTimeout: TimeInterval = 600

    /// A wait shorter than this is not worth a log line.
    static let loggedWaitSeconds: TimeInterval = 5

    /// True for an http(s) URL that the text path (`SidecarTextFetch`) does not take.
    static func appliesTo(_ url: URL, formatHint: String?) -> Bool {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            return false
        }
        return !SidecarTextFetch.appliesTo(url, formatHint: formatHint)
    }

    static func sessionConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = firstResponseTimeout
        configuration.timeoutIntervalForResource = firstResponseTimeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return configuration
    }

    /// Owned and process-wide, as `SidecarTextFetch.session` is, so a host's certificate decision
    /// reaches the probe too.
    static let session = URLSession(
        configuration: sessionConfiguration(), delegate: EngineTLS.sessionDelegate, delegateQueue: nil)

    /// Returns once the origin has answered with a success status; throws its status otherwise, or
    /// the transport error (a timeout after `firstResponseTimeout`). Cancelling the calling task ends
    /// the wait.
    static func wait(for url: URL, headers: [String: String], session: URLSession = session) async throws {
        var request = URLRequest(engineURL: url, timeoutInterval: firstResponseTimeout)
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        request.setValue("bytes=0-0", forHTTPHeaderField: "Range")
        let started = Date()
        let (bytes, response) = try await session.bytes(for: request)
        bytes.task.cancel()
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(status) else { throw SubtitleDecoderError.httpStatus(status) }
        let waited = Date().timeIntervalSince(started)
        if waited >= loggedWaitSeconds {
            EngineLog.emit("[Sidecar] origin answered after \(Int(waited)) s, reading \(url.lastPathComponent)")
        }
    }
}
