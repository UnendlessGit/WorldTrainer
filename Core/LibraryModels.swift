import Foundation

struct LibraryFolder: Codable, Hashable, Identifiable, Sendable {
    var id: UUID
    var name: String
    var createdAt: Date
}

enum RecordingUsability: String, Codable, Hashable, Sendable {
    case usable
    case unusable

    var label: String {
        switch self {
        case .usable: return "Usable"
        case .unusable: return "Unusable"
        }
    }
}

/// A catalog entry for one imported recording package.
struct LibraryRecording: Codable, Hashable, Identifiable, Sendable {
    var id: UUID
    var folderID: UUID?
    var directoryName: String
    var displayName: String
    var importedAt: Date
    var manifest: RecordingManifest?
    var issues: [RecordingIssue]
    var eventSummary: EventSummary?
    var packageBytes: Int64
    var videoWidth: Int?
    var videoHeight: Int?
    var videoDuration: Double?
    var sourcePackageName: String?

    var usability: RecordingUsability {
        issues.contains { $0.severity == .error } ? .unusable : .usable
    }

    var errorMessages: [String] {
        issues.filter { $0.severity == .error }.map(\.message)
    }

    var warningMessages: [String] {
        issues.filter { $0.severity == .warning }.map(\.message)
    }

    var duration: Double {
        manifest?.duration ?? videoDuration ?? 0
    }

    var pixelWidth: Int { manifest?.pixelWidth ?? videoWidth ?? 0 }
    var pixelHeight: Int { manifest?.pixelHeight ?? videoHeight ?? 0 }

    var capabilities: ActionCapabilities {
        eventSummary?.capabilities ?? ActionCapabilities()
    }

    var frameCount: Int64 {
        if let manifest { return manifest.encodedFrameCount }
        guard let videoDuration else { return 0 }
        return Int64(max(0, videoDuration * 30))
    }

    var eventCount: Int { eventSummary?.eventCount ?? manifest?.eventCount ?? 0 }
}

struct LibraryState: Codable, Sendable {
    var folders: [LibraryFolder] = []
    var recordings: [LibraryRecording] = []
}

struct RecordingPackageInspection: Sendable {
    var manifest: RecordingManifest?
    var issues: [RecordingIssue]
    var eventSummary: EventSummary?
    var packageBytes: Int64 = 0
    var videoWidth: Int?
    var videoHeight: Int?
    var videoDuration: Double?
    var suggestedName: String

    var isUsable: Bool { !issues.contains { $0.severity == .error } }
}
