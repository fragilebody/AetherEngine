import CoreVideo
import Foundation
import Metal
import Testing
@testable import AetherEngine

@Suite("Surface texture cache")
struct SurfaceTextureCacheTests {
    private func pool(_ format: OSType, width: Int = 64, height: Int = 32) -> CVPixelBufferPool? {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        return pool
    }

    private func buffer(from pool: CVPixelBufferPool) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        return buffer
    }

    @Test func aSurfaceKeepsItsTexturesAndEachPlaneHasItsOwn() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = SurfaceTextureCache(device: device)
        let frame = try #require(pool(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange).flatMap(buffer(from:)))

        let luma = try #require(cache.texture(frame, plane: 0, format: .r16Unorm))
        let chroma = try #require(cache.texture(frame, plane: 1, format: .rg16Unorm))
        #expect((luma.width, luma.height) == (64, 32))
        #expect((chroma.width, chroma.height) == (32, 16))
        #expect(cache.texture(frame, plane: 0, format: .r16Unorm) === luma)
        #expect(cache.texture(frame, plane: 1, format: .rg16Unorm) === chroma)
        // Another usage is another texture.
        #expect(cache.texture(frame, plane: 0, format: .r16Unorm, usage: [.shaderRead, .renderTarget]) !== luma)
        #expect(cache.count == 3)
    }

    /// A pool hands its surfaces out again once their buffers are released, and the texture comes back with them.
    @Test func aRecycledSurfaceFindsItsTexture() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = SurfaceTextureCache(device: device)
        let pool = try #require(pool(kCVPixelFormatType_32BGRA))
        var seen: [ObjectIdentifier: MTLTexture] = [:]
        var reused = 0
        for _ in 0..<20 {
            let frame = try #require(buffer(from: pool))
            let surface = try #require(CVPixelBufferGetIOSurface(frame)?.takeUnretainedValue())
            let texture = try #require(cache.texture(frame, format: .bgra8Unorm))
            if let earlier = seen[ObjectIdentifier(surface)] {
                #expect(earlier === texture)
                reused += 1
            }
            seen[ObjectIdentifier(surface)] = texture
        }
        #expect(reused > 0)
        #expect(cache.count == seen.count)
    }

    @Test func texturesNotAskedForAreLetGo() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = SurfaceTextureCache(device: device, maxAge: 0.05)
        let pool = try #require(pool(kCVPixelFormatType_32BGRA))
        let first = try #require(buffer(from: pool)), second = try #require(buffer(from: pool))
        _ = cache.texture(first, format: .bgra8Unorm)
        Thread.sleep(forTimeInterval: 0.1)
        _ = cache.texture(second, format: .bgra8Unorm)
        #expect(cache.count == 1)
        cache.removeAll()
        #expect(cache.count == 0)
    }
}
