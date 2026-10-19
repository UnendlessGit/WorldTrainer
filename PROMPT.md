# WorldTrainer — App Development Prompt

Build a native macOS app called **WorldTrainer** where I can import recordings, train world models locally, and interact with the generated worlds inside the same app.

## Project and repository

Work in `/Users/endless/Documents/WorldTrainer`.

Use this project's GitHub repository: **https://github.com/UnendlessGit/WorldTrainer**. Keep the app's source code and project documentation in that repository and use it for version control as development progresses.

This Markdown file is the editable product brief. Follow the intended functionality while choosing suitable architecture, frameworks, model design, and implementation details yourself.

## What the app should do

WorldTrainer should learn how an environment looks and responds to actions from recordings that include visual data and timed computer inputs. After training, I should be able to load the model, provide controls, and interact with its generated environment.

**Games are the main use case, but the app must not be tied to one game or limited to Minecraft.** It should be designed to support different games and other recorded visual environments with computer interactions, depending on the available training data and model capabilities.

For example, I could import Minecraft recordings, train a world model, and use WASD and mouse controls to explore a generated Minecraft-like world. **Minecraft is only an example**, not a special integration or a requirement to hard-code Minecraft-specific behavior. The same overall workflow should apply to other games and environments.

The Run experience must come from the trained model generating what happens next in response to my actions. It must not simply replay recordings or control the original application.

## Recording import

I have placed an example recording package in the project's `RecordingFormat/` folder. Inspect that example and support its recording format, including the video, timed inputs, and associated metadata. Preserve the relationship between what was visible and which actions happened at each time.

I already use separate recording software. **Do not build a recorder or recording controls into WorldTrainer.** Provide an Import button and the library functionality needed to use these recordings.

## Main tabs

The app should have five main tabs: **Library**, **World Models**, **Training**, **Run**, and **Settings**.

### Library

Let me create, rename, and delete folders and import recordings into a selected folder. Make it easy to browse, preview, organize, and remove recordings, see useful recording information, and identify recordings that cannot be used for training.

### World Models

Show a list of my world models and let me create, manage, and configure them independently. Each model should have its own dataset selections, model settings, training settings, and saved training progress or checkpoints.

Provide meaningful customization appropriate to the actual model implementation. Examples include resolution, color detail, temporal memory or context, model size, hidden layers, learning rate, and other useful training options. These are examples rather than a fixed technical design. Settings should have clear explanations and real effects.

### Training

Let me select a world model and start, pause, resume, and stop its training. Show useful live training statistics, progress, performance, and errors so I can understand what the model is doing and whether it is improving.

Training must be real, use my imported recordings and their actions, and produce saved models that can actually run in the app. Preserve work so training can continue later.

Short recording clips must not impose an artificial maximum generation duration. The design should account for ongoing generation, useful temporal memory, and consistency over time.

### Run

Let me select a trained world model and interact with its generated environment in a viewport inside this tab.

Support a broad range of computer actions, including key presses and holds, simultaneous keys, modifiers, relative mouse movement, mouse-button presses and holds, and scrolling. The usable controls should correspond to the recording data and what the model has learned, rather than assuming every model understands every action.

Provide convenient controls for starting, pausing, resetting, and stopping generation, along with useful runtime performance information. Input capture should behave naturally while interacting with the viewport and release cleanly when I leave it.

### Settings

Let me configure app-wide preferences, including appearance and where recordings, world models, and other app data are stored. Include useful storage, performance, and diagnostic options without making everyday use unnecessarily complicated.

## Local hardware, performance, and quality

My machine is an **Apple M3 Max MacBook Pro with 36 GB unified memory**. The app must train and run world models locally and use this hardware effectively.

Use suitable Apple-silicon acceleration, such as MLX and Metal where appropriate. Choose the implementation based on practical performance, memory usage, maintainability, and model quality.

Aim for a playable experience of approximately **15–30 generated FPS** with practical default settings on my machine. Measure the actual result rather than promising performance that has not been demonstrated.

I want strong visual quality, responsive controls, temporal consistency, and useful memory. Optimize carefully to preserve quality, and make unavoidable tradeoffs clear instead of silently reducing quality to reach a performance target.

## Overall expectations

Deliver a functional, cohesive macOS app with a clear interface and working import, training, model management, and interactive generation.

Choose the technical approach yourself. Keep the product broadly useful for games and other compatible environments. Be honest about demonstrated capabilities, hardware limitations, and any functionality that still needs work. Do not substitute mock training, simulated statistics, or recording playback for the real functionality.
