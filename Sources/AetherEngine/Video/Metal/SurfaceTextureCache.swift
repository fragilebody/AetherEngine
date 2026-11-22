import CoreVideo
import Foundation
import IOSurface
import Metal

/// Metal textures over the IOSurfaces of pixel buffers, kept per surface and reused (fork change).
///
/// `CVMetalTextureCache` made a new texture backing for every frame an `AVPlayerItemVideoOutput` or a
/// pixel buffer pool handed out, flushed or not: about 300 µs of CPU per 2560x1440 10-bit frame for its
/// two planes, against about 80 µs here, most of that the first sight of each surface. A pool or an
/// output cycles through a dozen or so surfaces, so after the first second every frame finds its
/// textures here.
///
/// A texture keeps its surface alive but doesn't hold the pixel buffer, so the pool still recycles the
/// surface: the caller keeps the buffer until the GPU is done with the frame, as with the CoreVideo
/// cache. A surface not asked for within `maxAge` is let go. Not thread-safe: use it under the
/// caller's lock or queue.
public final class SurfaceTextureCache {
    private struct Entry {
        let surface: IOSurfaceRef
        let plane: Int
        let format: MTLPixelFormat
        let usage: MTLTextureUsage
        let texture: MTLTexture
        var lastUsed: UInt64
    }

    public let device: MTLDevice
    private let maxAge: UInt64
    private var entries: [Entry] = []
    private var lastSweep: UInt64 = 0

    /// - Parameter maxAge: How long a texture is kept without being asked for, in seconds.
    public init(device: MTLDevice, maxAge: Double = 1) {
        self.device = device
        self.maxAge = UInt64(maxAge * 1_000_000_000)
    }

    /// Textures kept now.
    public var count: Int { entries.count }

    /// Plane `plane` of `buffer` (0 for a buffer that isn't planar) as a `format` texture with `usage`.
    /// nil when the buffer has no IOSurface or Metal refuses it.
    public func texture(_ buffer: CVPixelBuffer, plane: Int = 0, format: MTLPixelFormat,
                        usage: MTLTextureUsage = [.shaderRead]) -> MTLTexture? {
        guard let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() else { return nil }
        let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Old surfaces go at most a few times per `maxAge`, not on every call.
        if now &- lastSweep > maxAge / 4 {
            lastSweep = now
            entries.removeAll { now &- $0.lastUsed > maxAge }
        }
        if let i = entries.firstIndex(where: { $0.surface === surface && $0.plane == plane && $0.format == format
                                               && $0.usage == usage }) {
            entries[i].lastUsed = now
            return entries[i].texture
        }
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
        guard width > 0, height > 0 else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = usage
        guard let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: plane) else { return nil }
        entries.append(Entry(surface: surface, plane: plane, format: format, usage: usage, texture: texture, lastUsed: now))
        return texture
    }

    /// Lets go of every texture (a pool that changed size will not hand its old surfaces out again).
    public func removeAll() { entries.removeAll() }
}
