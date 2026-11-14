/*
 * The blue-noise ("fruit") generator below is a Swift port of mpv's video/out/dither.c:
 *
 * Generate a dithering matrix for downsampling images.
 *
 * Copyright © 2013  Wessel Dankers <wsl@fruit.je>
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 */

import Foundation

/// The threshold matrices `VideoDitherer` tiles over the picture. Every matrix is square, row-major, and
/// holds each rank `k / size²` for `k` in `0 ..< size²` exactly once, so its mean is `(size² - 1) / (2 size²)`.
enum DitherMatrix {

    /// Sizes `fruit(sizeLog2:)` accepts, as mpv's `--dither-size-fruit` does. 8 (256x256) takes seconds to build.
    static let fruitSizeLog2Range = 2...8

    /// mpv's default blue-noise matrix (`--dither=fruit`), `2^sizeLog2` on a side. Cells are ranked one at a
    /// time: each new cell is the one with the least accumulated Gaussian energy from the cells ranked
    /// before it, so consecutive ranks land far apart and the pattern has no low-frequency structure.
    static func fruit(sizeLog2: Int) -> [Float] {
        let sizeb = min(max(sizeLog2, fruitSizeLog2Range.lowerBound), fruitSizeLog2Range.upperBound)
        let size = 1 << sizeb
        let size2 = size * size
        let radius = size / 2 - 1
        let middle = radius | (radius << sizeb)
        let gaussSize = radius * 2 + 1
        let gaussSize2 = Double(gaussSize * gaussSize)

        var gauss = [UInt64](repeating: 0, count: size2)
        let sigma = -log(1.5 / Double(UInt64.max) * gaussSize2) / Double(radius)
        func xy(_ x: Int, _ y: Int) -> Int { x | (y << sizeb) }
        for gy in 0...radius {
            for gx in 0...gy {
                let cx = gx - radius, cy = gy - radius
                let e = exp(-(Double(cx * cx + cy * cy)).squareRoot() * sigma)
                let scaled = e / gaussSize2 * Double(UInt64.max)
                let v = scaled >= Double(UInt64.max) ? UInt64.max : UInt64(max(0, scaled))
                let fx = gaussSize - 1 - gx, fy = gaussSize - 1 - gy
                for index in [xy(gx, gy), xy(gy, gx), xy(gx, fy), xy(gy, fx), xy(fx, gy), xy(fy, gx), xy(fx, fy), xy(fy, fx)] {
                    gauss[index] = v
                }
            }
        }

        // Raw buffers: this loop runs size2 * size2 * 2 steps (33 million at the default size), and checked
        // array access makes an unoptimised build of it take seconds.
        let energy = UnsafeMutablePointer<UInt64>.allocate(capacity: size2)
        let ranked = UnsafeMutablePointer<Bool>.allocate(capacity: size2)
        let rank = UnsafeMutablePointer<Int>.allocate(capacity: size2)
        let ties = UnsafeMutablePointer<Int>.allocate(capacity: size2)
        let kernel = UnsafeMutablePointer<UInt64>.allocate(capacity: size2)
        defer { energy.deallocate(); ranked.deallocate(); rank.deallocate(); ties.deallocate(); kernel.deallocate() }
        energy.initialize(repeating: 0, count: size2)
        ranked.initialize(repeating: false, count: size2)
        rank.initialize(repeating: 0, count: size2)
        ties.initialize(repeating: 0, count: size2)
        for index in 0..<size2 { kernel[index] = gauss[index] }

        // mpv draws its tie-breaks from av_lfg seeded with 123. Any fixed sequence gives a matrix of the same
        // quality; this one keeps the generator free of FFmpeg.
        var random: UInt64 = 123
        for k in 0..<size2 {
            // The unranked cell with the least energy; ties are broken at random.
            var least = UInt64.max
            var count = 0
            var c = 0
            while c < size2 {
                if !ranked[c] {
                    let total = energy[c]
                    if total <= least {
                        if total != least {
                            least = total
                            count = 0
                        }
                        ties[count] = c
                        count += 1
                    }
                }
                c += 1
            }
            let cell: Int
            if count == 1 {
                cell = ties[0]
            } else if count == size2 {
                cell = size2 / 2
            } else {
                random &+= 0x9E37_79B9_7F4A_7C15
                var z = random
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                z ^= z >> 31
                cell = ties[Int(z % UInt64(count))]
            }
            // Add the Gaussian centred on that cell, wrapping at the edges so the matrix tiles.
            ranked[cell] = true
            let start = (middle + size2 - cell) & (size2 - 1)
            var m = 0
            var g = start
            while g < size2 {
                energy[m] = energy[m] &+ kernel[g]
                m += 1
                g += 1
            }
            g = 0
            while m < size2 {
                energy[m] = energy[m] &+ kernel[g]
                m += 1
                g += 1
            }
            rank[cell] = k
        }
        return (0..<size2).map { Float(rank[$0]) / Float(size2) }
    }

    /// The Bayer ordered-dither matrix (`--dither=ordered`), `size` a power of two on a side.
    static func bayer(size: Int) -> [Float] {
        var bits = 0
        while (1 << bits) < max(size, 2) { bits += 1 }
        let side = 1 << bits
        var matrix = [Float](repeating: 0, count: side * side)
        for y in 0..<side {
            for x in 0..<side {
                // Bit-reversed interleave of (x ^ y, y): the classic recursive construction, unrolled.
                var value = 0
                var a = x ^ y, b = y
                for _ in 0..<bits {
                    value = (value << 2) | ((a & 1) << 1) | (b & 1)
                    a >>= 1
                    b >>= 1
                }
                matrix[y * side + x] = Float(value) / Float(side * side)
            }
        }
        return matrix
    }

    /// The side of the ordered matrix, libplacebo's (which mpv's default `vo=gpu-next` uses).
    static let orderedSize = 16

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var fruitCache: [Int: [Float]] = [:]

    /// The fruit matrix of this size if the process has built it already.
    static func builtFruit(sizeLog2: Int) -> [Float]? {
        cacheLock.lock(); defer { cacheLock.unlock() }
        return fruitCache[sizeLog2]
    }

    /// `fruit(sizeLog2:)`, built once per size for the process: the 64x64 default costs about 50 ms
    /// optimised and several times that in a debug build, so callers on a playback path build it elsewhere.
    static func cachedFruit(sizeLog2: Int) -> [Float] {
        if let cached = builtFruit(sizeLog2: sizeLog2) { return cached }
        let built = fruit(sizeLog2: sizeLog2)
        cacheLock.lock(); defer { cacheLock.unlock() }
        fruitCache[sizeLog2] = built
        return built
    }

    /// Where `cachedFruit` runs when a pass must not wait for it.
    static let buildQueue = DispatchQueue(label: "aether.dither.matrix", qos: .userInitiated)
}
