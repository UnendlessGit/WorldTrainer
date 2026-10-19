import AVFoundation
import Foundation

/// Validates one recording package without modifying it.
enum RecordingInspector {
    static func inspect(packageDirectory: URL) async -> RecordingPackageInspection {
        let manager = FileManager.default
        var issues: [RecordingIssue] = []
        var manifest: RecordingManifest?
        let suggestedName = packageDirectory.lastPathComponent

        let manifestURL = packageDirectory.appendingPathComponent("manifest.json")
        if !manager.fileExists(atPath: manifestURL.path) {
            issues.append(.error("missing-manifest", "manifest.json is missing."))
        } else {
            do {
                let data = try Data(contentsOf: manifestURL)
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let decoded = try decoder.decode(RecordingManifest.self, from: data)
                manifest = decoded
                if !decoded.isStructurallyValid {
                    issues.append(.error("invalid-manifest", "manifest.json is missing required fields or contains out-of-range values."))
                }
            } catch {
                issues.append(.error("invalid-manifest", "manifest.json could not be decoded: \(error.localizedDescription)"))
            }
        }

        var summary: EventSummary?
        if let manifest {
            let eventURL = packageDirectory.appendingPathComponent(manifest.eventFile)
            if !manager.fileExists(atPath: eventURL.path) {
                issues.append(.error("missing-events", "The input event file \(manifest.eventFile) is missing."))
            } else {
                do {
                    summary = try EventFile.summarize(url: eventURL, expectedCount: manifest.eventCount)
                } catch {
                    issues.append(.error("invalid-events", error.localizedDescription))
                }
            }
        }

        var videoWidth: Int?
        var videoHeight: Int?
        var videoDuration: Double?
        if let manifest {
            let videoURL = packageDirectory.appendingPathComponent(manifest.videoFile)
            if !manager.fileExists(atPath: videoURL.path) {
                issues.append(.error("missing-video", "The video file \(manifest.videoFile) is missing."))
            } else {
                do {
                    let info = try await VideoFrames.loadTrack(url: videoURL)
                    videoDuration = info.duration
                    let size = info.displaySize
                    videoWidth = Int(size.width.rounded())
                    videoHeight = Int(size.height.rounded())
                    if info.duration < 0.05 {
                        issues.append(.error("empty-video", "The video track has no usable duration."))
                    }
                    if let width = videoWidth, let height = videoHeight {
                        if abs(width - manifest.pixelWidth) > 2 || abs(height - manifest.pixelHeight) > 2 {
                            issues.append(.warning("dimension-mismatch",
                                "The manifest reports \(manifest.pixelWidth)×\(manifest.pixelHeight) but the video displays \(width)×\(height). The video dimensions will be used."))
                        }
                    }
                    if manifest.deliveredFPS > 0, info.nominalFrameRate > 0 {
                        let ratio = info.nominalFrameRate / Float(manifest.deliveredFPS)
                        if ratio < 0.5 || ratio > 2 {
                            issues.append(.warning("fps-mismatch",
                                "The manifest reports \(manifest.deliveredFPS.formatted(.number.precision(.fractionLength(1)))) FPS but the video is \(Double(info.nominalFrameRate).formatted(.number.precision(.fractionLength(1)))) FPS."))
                        }
                    }
                } catch {
                    issues.append(.error("unreadable-video", error.localizedDescription))
                }
            }
            if let thumbnail = manifest.thumbnailFile {
                let thumbnailURL = packageDirectory.appendingPathComponent(thumbnail)
                if !manager.fileExists(atPath: thumbnailURL.path) {
                    issues.append(.warning("missing-thumbnail", "The thumbnail is missing; WorldTrainer will generate one."))
                }
            }
        }

        if let manifest, summary == nil, !issues.contains(where: { $0.code == "invalid-events" || $0.code == "missing-events" }) {
            issues.append(.warning("no-events", "No input events could be summarized."))
        }
        if let summary, summary.eventCount == 0 {
            issues.append(.warning("empty-events", "The recording contains no input events. It can train video but cannot teach controls."))
        }

        return RecordingPackageInspection(
            manifest: manifest,
            issues: issues,
            eventSummary: summary,
            packageBytes: AppDirectories.directoryBytes(packageDirectory),
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            videoDuration: videoDuration,
            suggestedName: manifest?.name ?? suggestedName)
    }

    /// A quick structural check used before copying a folder.
    static func looksLikePackage(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path)
            || FileManager.default.fileExists(atPath: directory.appendingPathComponent("events.atrevents").path)
            || FileManager.default.fileExists(atPath: directory.appendingPathComponent("capture.mov").path)
    }
}
