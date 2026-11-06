# Fork notes: fragilebody/AetherEngine

This repository is a public fork of
[superuser404notfound/AetherEngine](https://github.com/superuser404notfound/AetherEngine), kept
public in source form as the LGPL requires. It carries one change on top of upstream, on the branch
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

Upstream tag `7.58.0`. Fork tag `7.58.0-memory.1`.

## Rebasing onto a new upstream tag

```bash
git remote add upstream https://github.com/superuser404notfound/AetherEngine.git   # once
git remote set-url --push upstream DISABLED                                          # once: never push there
git fetch upstream --tags
git checkout memory-segments
git rebase --onto <new-tag> <old-tag> memory-segments    # e.g. --onto 7.59.0 7.58.0
swift build --build-tests
swift test --skip-build --filter 'SegmentMemoryStoreTests|SegmentCacheTests|ProgressiveSegmentDeliveryTests|SegmentRetentionTests|SessionCacheContractTests'
git tag <new-tag>-memory.1
git push --force-with-lease origin memory-segments
git push origin <new-tag>-memory.1
```

Conflicts land almost always in `SegmentCache.swift` (entries are `SegmentStorage`, not `URL`) and in
`MP4SegmentMuxer.swift` (the staging is `SegmentStaging`). Then bump the host app's exact package
version to the new fork tag.

## Quiet logging

Engine log lines no longer go to OSLog (and so the Xcode/terminal console) unless the `AETHER_LOG`
environment variable is set to a value other than `0`. Host handlers (`EngineLog.handler`) still get
every `.info` line. See `EngineLog.osLogEnabled`.
