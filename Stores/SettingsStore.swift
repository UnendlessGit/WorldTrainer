import AppKit
import Foundation
import SwiftUI

enum AppAppearance: String, Codable, CaseIterable, Identifiable, Sendable {
    case system
    case light
    case dark

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
}

struct AppPreferences: Codable, Sendable {
    var appearance: AppAppearance = .system
    var dataRootPath: String = AppDirectories.defaultRoot.path
    var memoryLimitGB: Double = 24
    var cacheLimitGB: Double = 4
    var defaultRunFPS: Double = 20
    var runWarmupFrames: Int = 24
    var showAdvancedMetrics = true
    var confirmBeforeDelete = true
    var datasetCacheLimitGB: Double = 40
    var keepBestCheckpointsOnly = false

    var dataRoot: URL {
        URL(fileURLWithPath: (dataRootPath as NSString).expandingTildeInPath, isDirectory: true)
    }
}

@MainActor
final class SettingsStore: ObservableObject {
    @Published var preferences: AppPreferences {
        didSet { save() }
    }

    private let defaultsKey = "WorldTrainer.Preferences.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: defaultsKey),
           let decoded = try? JSONDecoder().decode(AppPreferences.self, from: data) {
            preferences = decoded
        } else {
            preferences = AppPreferences()
        }
    }

    var directories: AppDirectories {
        AppDirectories(root: preferences.dataRoot)
    }

    func save() {
        if let data = try? JSONEncoder().encode(preferences) {
            UserDefaults.standard.set(data, forKey: defaultsKey)
        }
    }

    func reset() {
        preferences = AppPreferences()
    }

    func storageBreakdown() -> [(String, Int64)] {
        let directories = self.directories
        return [
            ("Recordings", AppDirectories.directoryBytes(directories.recordingsDirectory)),
            ("Models & checkpoints", AppDirectories.directoryBytes(directories.modelsDirectory)),
            ("Training caches", AppDirectories.directoryBytes(directories.trainingCacheDirectory)),
            ("Total", AppDirectories.directoryBytes(directories.root)),
        ]
    }

    func clearTrainingCaches() throws {
        let directory = directories.trainingCacheDirectory
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
}
