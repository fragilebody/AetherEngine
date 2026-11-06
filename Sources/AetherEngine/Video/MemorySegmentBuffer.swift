import Darwin
import Foundation

// Fork (memory-segments): in-memory storage for loopback HLS segments.
//
// Upstream writes every remuxed segment to a staging file under NSTemporaryDirectory() and serves it
// mmapped, so every byte watched is written to disk once (about 90 MB per minute at 12 Mbit/s). With
// `LoadOptions.segmentMemoryBudgetBytes` set, `SegmentCache` hands the muxer one of these buffers
// instead of a staging file while the bytes it holds in memory fit the budget.

/// One purgeable VM region (`vm_allocate` with `VM_FLAGS_PURGABLE`). Volatile regions may be emptied
/// by the kernel under memory pressure; a region is only read while it is non-volatile.
final class PurgeableRegion: @unchecked Sendable {
    let address: vm_address_t
    let capacity: Int

    init?(capacity requested: Int) {
        let page = Int(getpagesize())
        let rounded = max(page, (requested + page - 1) / page * page)
        var address: vm_address_t = 0
        let kr = vm_allocate(mach_task_self_, &address, vm_size_t(rounded),
                             VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE)
        guard kr == KERN_SUCCESS, address != 0 else { return nil }
        self.address = address
        self.capacity = rounded
    }

    deinit {
        vm_deallocate(mach_task_self_, address, vm_size_t(capacity))
    }

    var base: UnsafeMutableRawPointer { UnsafeMutableRawPointer(bitPattern: UInt(address))! }

    /// Sets the purgeable state and returns the state it replaced, or nil when the call failed.
    func setState(_ state: Int32) -> Int32? {
        var value = state
        let kr = vm_purgable_control(mach_task_self_, address, VM_PURGABLE_SET_STATE, &value)
        return kr == KERN_SUCCESS ? value : nil
    }

    func state() -> Int32? {
        var value: Int32 = 0
        let kr = vm_purgable_control(mach_task_self_, address, VM_PURGABLE_GET_STATE, &value)
        return kr == KERN_SUCCESS ? value : nil
    }
}

/// Bytes held by every live `MemorySegmentBuffer` of one cache: entries, the segment the muxer is
/// writing, and evicted buffers a serve still holds. Admission measures against this, so what it
/// counts is what is actually in RAM rather than what the cache still lists.
final class MemorySegmentLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var _residentBytes = 0

    var residentBytes: Int { lock.withLock { _residentBytes } }

    func adjust(by delta: Int) {
        guard delta != 0 else { return }
        lock.withLock { _residentBytes += delta }
    }
}

/// A segment held in memory. The muxer appends to it like it appended to a staging file; the cache
/// adopts it as it stands; serves get a `Data` over its bytes without a copy.
///
/// Purgeable rules: a buffer is non-volatile while it is written and while it sits at or ahead of the
/// consumer's fetch target (AVPlayer has not fetched it yet). Behind the target it is watched history
/// and the cache marks it purgeable; macOS may then reclaim it. Every read pins the region first
/// (non-volatile until the last reader lets go), and a pin that finds the region emptied reports the
/// buffer as purged, which the cache treats exactly like a segment file that vanished (AE#451).
final class MemorySegmentBuffer: @unchecked Sendable {
    private static let idLock = NSLock()
    nonisolated(unsafe) private static var nextID: UInt64 = 0  // guarded by idLock

    static let minimumCapacity = 1 << 20

    /// Unique for the process, so a sealed record keyed by it can never match a later buffer.
    let id: UInt64
    let index: Int

    private let lock = NSLock()
    private var region: PurgeableRegion?
    private var _length = 0
    private var pins = 0
    private var wantsPurgeable = false
    private var isVolatile = false
    private var _purged = false
    private let ledger: MemorySegmentLedger?

    init?(index: Int, initialCapacity: Int = MemorySegmentBuffer.minimumCapacity,
          ledger: MemorySegmentLedger? = nil) {
        guard let region = PurgeableRegion(capacity: max(Self.minimumCapacity, initialCapacity)) else {
            return nil
        }
        Self.idLock.lock()
        Self.nextID &+= 1
        self.id = Self.nextID
        Self.idLock.unlock()
        self.index = index
        self.region = region
        self.ledger = ledger
    }

    deinit {
        if !_purged { ledger?.adjust(by: -_length) }
    }

    var length: Int { lock.withLock { _length } }

    /// Whether the region is volatile right now, i.e. the kernel may reclaim it.
    var isMarkedPurgeable: Bool { lock.withLock { isVolatile && !_purged } }

    /// Appends `count` bytes, growing the region when it is full. A grown region replaces the old
    /// one; a serve still holding the old one keeps it alive, so the bytes it reads never move.
    /// False when memory could not be allocated or the buffer was purged.
    func append(_ bytes: UnsafeRawPointer, count: Int) -> Bool {
        guard count > 0 else { return true }
        lock.lock()
        defer { lock.unlock() }
        guard !_purged, let current = region else { return false }
        if _length + count > current.capacity {
            var newCapacity = current.capacity
            while newCapacity < _length + count { newCapacity *= 2 }
            guard let grown = PurgeableRegion(capacity: newCapacity) else { return false }
            if _length > 0 { memcpy(grown.base, current.base, _length) }
            if isVolatile { _ = grown.setState(VM_PURGABLE_VOLATILE) }
            region = grown
        }
        memcpy(region!.base.advanced(by: _length), bytes, count)
        _length += count
        ledger?.adjust(by: count)
        return true
    }

    /// Drops every byte written so far (the muxer's AE#222 moov prime fragment).
    func truncate() {
        lock.lock()
        let dropped = _purged ? 0 : _length
        _length = 0
        lock.unlock()
        ledger?.adjust(by: -dropped)
    }

    /// True once the kernel emptied the region. Only a volatile region can be emptied, so this asks
    /// the kernel only then.
    var isPurged: Bool {
        lock.lock()
        defer { lock.unlock() }
        if !_purged, isVolatile, let region, region.state() == VM_PURGABLE_EMPTY {
            markPurgedLocked()
        }
        return _purged
    }

    /// Marks the buffer purgeable (watched history) or not (still ahead of the consumer). False when
    /// making it non-volatile found it already purged.
    @discardableResult
    func setPurgeable(_ purgeable: Bool) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !_purged else { return false }
        wantsPurgeable = purgeable
        if purgeable {
            if pins == 0, !isVolatile, let region {
                _ = region.setState(VM_PURGABLE_VOLATILE)
                isVolatile = true
            }
            return true
        }
        return makeNonVolatileLocked()
    }

    /// The whole buffer as `Data`, without a copy. The region stays pinned until the `Data` (and
    /// every copy of it) is released. nil when the buffer was purged.
    func pinnedData() -> Data? {
        lock.lock()
        guard let region = pinLocked() else {
            lock.unlock()
            return nil
        }
        let count = _length
        lock.unlock()
        guard count > 0 else {
            unpin(region)
            return Data()
        }
        return Data(bytesNoCopy: region.base, count: count, deallocator: .custom { [self] _, _ in
            self.unpin(region)
        })
    }

    /// Up to `maxBytes` copied from `offset`, for a progressive reader. Empty when nothing past
    /// `offset` is written yet, nil when the buffer was purged.
    func copyBytes(from offset: Int, maxBytes: Int) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard offset < _length else { return _purged ? nil : Data() }
        guard let region = pinLocked() else { return nil }
        let count = min(maxBytes, _length - offset)
        let data = Data(bytes: region.base.advanced(by: offset), count: count)
        unpinLocked(region)
        return data
    }

    /// Tests only: empties the region as the kernel would under memory pressure.
    func purgeForTesting() {
        lock.lock()
        if let region, !_purged {
            _ = region.setState(VM_PURGABLE_EMPTY)
            isVolatile = true
        }
        lock.unlock()
    }

    // MARK: - Locked helpers

    private func pinLocked() -> PurgeableRegion? {
        guard !_purged, let region else { return nil }
        if isVolatile, !makeNonVolatileLocked() { return nil }
        pins += 1
        return region
    }

    private func unpin(_ pinned: PurgeableRegion) {
        lock.lock()
        unpinLocked(pinned)
        lock.unlock()
    }

    private func unpinLocked(_ pinned: PurgeableRegion) {
        pins -= 1
        if pins == 0, wantsPurgeable, !isVolatile, !_purged, let region, region === pinned {
            _ = region.setState(VM_PURGABLE_VOLATILE)
            isVolatile = true
        }
    }

    private func makeNonVolatileLocked() -> Bool {
        guard isVolatile, let region else { return !_purged }
        let previous = region.setState(VM_PURGABLE_NONVOLATILE)
        isVolatile = false
        if previous == VM_PURGABLE_EMPTY {
            markPurgedLocked()
            return false
        }
        return true
    }

    private func markPurgedLocked() {
        guard !_purged else { return }
        _purged = true
        ledger?.adjust(by: -_length)
    }
}

/// Where a segment the producer is writing lives: a staging file (upstream) or a memory buffer.
enum SegmentStaging: Equatable, @unchecked Sendable {
    case file(URL)
    case memory(MemorySegmentBuffer)

    static func == (lhs: SegmentStaging, rhs: SegmentStaging) -> Bool {
        switch (lhs, rhs) {
        case (.file(let a), .file(let b)): return a == b
        case (.memory(let a), .memory(let b)): return a === b
        default: return false
        }
    }

    /// Identity that holds no reference to the buffer, for records that outlive it.
    var key: Key {
        switch self {
        case .file(let url): return .file(url)
        case .memory(let buffer): return .memory(buffer.id)
        }
    }

    enum Key: Hashable {
        case file(URL)
        case memory(UInt64)
    }

    var fileURL: URL? {
        if case .file(let url) = self { return url }
        return nil
    }
}
