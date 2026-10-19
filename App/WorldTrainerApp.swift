import SwiftUI

@main
struct WorldTrainerApp: App {
    @StateObject private var store = AppStore()

    init() {
        if CommandLine.arguments.contains("--self-test") {
            SelfTest.runAndExit()
        }
    }

    var body: some Scene {
        Window("WorldTrainer", id: "main") {
            RootView()
                .environmentObject(store)
                .environmentObject(store.library)
                .environmentObject(store.models)
                .environmentObject(store.training)
                .environmentObject(store.run)
                .environmentObject(store.settings)
                .preferredColorScheme(store.settings.preferences.appearance.colorScheme)
                .frame(minWidth: 1100, minHeight: 740)
        }
        .windowToolbarStyle(.unified(showsTitle: true))
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Run") {
                Button("Start or Restart Generation") {
                    guard let id = store.runModelID ?? store.models.trainedModels.first?.id else { return }
                    store.runModelID = id
                    Task { await store.startRun(modelID: id) }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])

                Button("Stop Generation") {
                    store.run.stop()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!store.run.isActive)

                Button("Release All Input") {
                    store.run.capture.releaseAll()
                }
                .keyboardShortcut("k", modifiers: [.command, .shift])
            }
            CommandMenu("Training") {
                Button("Start Training") {
                    guard let id = store.trainingModelID else { return }
                    _ = store.startTraining(modelID: id, resume: false)
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
                .disabled(store.training.isRunning)

                Button("Pause Training") {
                    store.training.pause()
                }
                .disabled(!store.training.canPause)

                Button("Stop Training") {
                    store.training.cancel()
                }
                .disabled(!store.training.isRunning)
            }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        TabView(selection: Binding(
            get: { store.selectedTab },
            set: { store.selectTab($0) }
        )) {
            LibraryView()
                .tabItem { Label(AppTab.library.title, systemImage: AppTab.library.systemImage) }
                .tag(AppTab.library)
            ModelsView()
                .tabItem { Label(AppTab.models.title, systemImage: AppTab.models.systemImage) }
                .tag(AppTab.models)
            TrainingView()
                .tabItem { Label(AppTab.training.title, systemImage: AppTab.training.systemImage) }
                .tag(AppTab.training)
            RunView()
                .tabItem { Label(AppTab.run.title, systemImage: AppTab.run.systemImage) }
                .tag(AppTab.run)
            SettingsView()
                .tabItem { Label(AppTab.settings.title, systemImage: AppTab.settings.systemImage) }
                .tag(AppTab.settings)
        }
        .padding(.top, 4)
    }
}
