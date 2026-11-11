# Fork notes: fragilebody/AetherEngine

This repository is a public fork of
[superuser404notfound/AetherEngine](https://github.com/superuser404notfound/AetherEngine), kept
public in source form as the LGPL requires. It carries a few changes on top of upstream, on the branch
`memory-segments`. Nothing here is sent upstream.

## What differs from upstream

**In-memory loopback segments (`LoadOptions.segmentMemoryBudgetBytes`).** Upstream writes every
remuxed HLS-fMP4 segment of the native loopback route to
`NSTemporaryDirectory()/aether-segments/<session>/` and serves it mmapped. With a byte budget set:

- `SegmentCache.admitMemorySegment(index:)` decides per segment, when the muxer opens it, whether it
  fits in memory: bytes resident in memory buffers plus an estimate (the last segment's size) must fit
  the budget, after evicting memory history outside the hard window, farthest from the target first.
  Otherwise the segment is staged in a file exactly as upstream. A segment never moves between stores.
- `MP4SegmentMuxer` takes a `memoryStaging` closure and appends fragments to a `MemorySegmentBuffer`
  (`Sources/AetherEngine/Video/MemorySegmentBuffer.swift`) instead of a staging file descriptor.
- The buffer is a `vm_allocate(VM_FLAGS_PURGABLE)` region. It is non-volatile while written and while
  at or ahead of the consumer's fetch target; behind the target it is volatile (purgeable). Every read
  pins it; a pin that finds it emptied drops the entry like a vanished file (AE#451), and the provider
  produces the segment again.
- Serves return `Data(bytesNoCopy:)` over the pinned buffer; the sendfile fast path stays file-only.
  Progressive delivery reads the growing buffer.
- Memory bytes are budgeted apart from the disk retention budget (`sessionCacheByteBudget`), which keeps
  counting disk bytes only.
- nil budget (the default) is upstream behaviour unchanged. Live sessions always use disk.

Still written to disk with a budget set: segments that spill, the I-frame rendition and scrub-preview
payload caches (`iframes/`, `preview/` under the session directory, only when those features run),
and the software path's packet cache (its own `softwarePacketCacheStorage` option).

Files touched: `SegmentCache.swift`, `MemorySegmentBuffer.swift` (new), `MP4SegmentMuxer.swift`,
`HLSSegmentProducer.swift`, `VideoSegmentProvider.swift`, `HLSVideoEngine.swift`,
`CompositorPreviewBuilder.swift`, `IFrameFragmentBuilder.swift`, `PlayerState.swift`,
`AetherEngine+Loading.swift`, `AetherEngine+ReloadWithOptions.swift`, `docs/api.md`, tests in
`Tests/AetherEngineTests/SegmentMemoryStoreTests.swift` (plus `CutOutcome` pattern updates in a few
existing muxer tests).

## Based on

Upstream tag `7.58.0`. Fork tag `7.58.0-memory.3`.

## Rebasing onto a new upstream tag

```bash
git remote add upstream https://github.com/superuser404notfound/AetherEngine.git   # once
git remote set-url --push upstream DISABLED                                          # once: never push there
git fetch upstream --tags
git checkout memory-segments
git rebase --onto <new-tag> <old-tag> memory-segments    # e.g. --onto 7.59.0 7.58.0
swift build --build-tests
swift test --skip-build --filter 'SegmentMemoryStoreTests|SegmentCacheTests|ProgressiveSegmentDeliveryTests|SegmentRetentionTests|SessionCacheContractTests|DitherMatrixTests|VideoDitherOptionsTests|VideoDitherPassTests'
git tag <new-tag>-memory.1
git push --force-with-lease origin memory-segments
git push origin <new-tag>-memory.1
```

Conflicts land almost always in `SegmentCache.swift` (entries are `SegmentStorage`, not `URL`) and in
`MP4SegmentMuxer.swift` (the staging is `SegmentStaging`). Then bump the host app's exact package
version to the new fork tag.

## Dithering

**mpv-style dithering of SDR video (`LoadOptions.videoDither`, public `VideoDitherer`).** Upstream hands
decoded Y'CbCr to the display layer and leaves the conversion to RGB, and its rounding, to the system.
With the option set:

- `VideoDitherer` (`Sources/AetherEngine/Video/VideoDitherer.swift`) converts a bi-planar Y'CbCr frame
  (8 or 10 bit) to R'G'B' in float and quantises it to 8 or 10 bits through a threshold matrix tiled
  over the picture, in one Metal render pass (`Video/Metal/DitherShaderSource.swift`). The formula is
  mpv's: `floor(colour * (2^depth - 1) + threshold) / (2^depth - 1)`.
- `DitherMatrix` (`Video/DitherMatrix.swift`) is a Swift port of mpv's `video/out/dither.c` (the "fruit"
  blue-noise generator, LGPL 2.1+, copyright notice kept) plus a Bayer matrix.
- The output is RGB: `kCVPixelFormatType_ARGB2101010LEPacked` at 10 bits (the default, `.auto`),
  `kCVPixelFormatType_32BGRA` at 8.
- The software host wraps its frame handler (`SoftwarePlaybackHost.ditheredFrameHandler`,
  `Decoder/DitheringFrameHandler.swift`), after the upscale stage. HDR, Dolby Vision and the dual-layer
  route are skipped.
- The native route is untouched. The host app's own compositor, which already takes AVPlayer's frames,
  calls `VideoDitherer` there.
- nil (the default) is upstream behaviour unchanged.

Files touched: `PlayerState.swift`, `AetherEngine+Loading.swift`, `AetherEngine+ReloadWithOptions.swift`,
`SoftwarePlaybackHost.swift`, `docs/api.md`, the new files above, and `aetherctl` (`play --dither`,
`Sources/aetherctl/main.swift`, `PlaybackCmd.swift`, `docs/cli.md`); tests in
`Tests/AetherEngineTests/VideoDitherTests.swift`.

## Quiet logging

Engine log lines no longer go to OSLog (and so the Xcode/terminal console) unless the `AETHER_LOG`
environment variable is set to a value other than `0`. Host handlers (`EngineLog.handler`) still get
every `.info` line. See `EngineLog.osLogEnabled`.
