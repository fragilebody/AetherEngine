import CoreMedia
import CoreVideo

/// Puts the dither pass (`VideoDitherer`) in front of a decoded-frame handler. `dither` returns nil for a
/// frame it does not take (HDR, an unsupported format, a GPU failure); that frame is delivered as decoded.
enum DitheringFrameHandler {
    static func wrap(
        _ handler: @escaping DecodedFrameHandler,
        dither: @escaping @Sendable (CVPixelBuffer) -> CVPixelBuffer?
    ) -> DecodedFrameHandler {
        { buffer, pts, hdr10Plus in
            handler(dither(buffer) ?? buffer, pts, hdr10Plus)
        }
    }
}
