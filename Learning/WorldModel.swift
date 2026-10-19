import Foundation
import MLX
import MLXNN

final class ConvNormActivation: Module {
    let conv: Conv2d
    let norm: GroupNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int = 3, stride: Int = 1, padding: Int = 1) {
        conv = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding))
        norm = GroupNorm(groupCount: Self.groups(for: outputChannels), dimensions: outputChannels)
    }

    static func groups(for channels: Int) -> Int {
        var groups = min(8, max(1, channels / 4))
        while groups > 1 && channels % groups != 0 { groups -= 1 }
        return groups
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(norm(conv(x)))
    }
}

/// Convolutional GRU used as the persistent temporal memory of the world model.
/// The hidden tensor survives for the whole Run session; nothing in the runtime
/// caps how long it may be carried.
final class ConvGRUCell: Module {
    let inputUpdate: Conv2d
    let hiddenUpdate: Conv2d
    let inputCandidate: Conv2d
    let hiddenCandidate: Conv2d

    init(inputChannels: Int, hiddenChannels: Int) {
        inputUpdate = Conv2d(inputChannels: inputChannels, outputChannels: hiddenChannels * 2,
                             kernelSize: IntOrPair(3), padding: IntOrPair(1))
        hiddenUpdate = Conv2d(inputChannels: hiddenChannels, outputChannels: hiddenChannels * 2,
                              kernelSize: IntOrPair(3), padding: IntOrPair(1))
        inputCandidate = Conv2d(inputChannels: inputChannels, outputChannels: hiddenChannels,
                                kernelSize: IntOrPair(3), padding: IntOrPair(1))
        hiddenCandidate = Conv2d(inputChannels: hiddenChannels, outputChannels: hiddenChannels,
                                 kernelSize: IntOrPair(3), padding: IntOrPair(1))
    }

    func callAsFunction(_ input: MLXArray, hidden: MLXArray) -> MLXArray {
        let update = sigmoid(inputUpdate(input) + hiddenUpdate(hidden))
        let parts = split(update, parts: 2, axis: -1)
        let reset = parts[0]
        let gate = parts[1]
        let candidate = tanh(inputCandidate(input) + hiddenCandidate(reset * hidden))
        return (1 - gate) * hidden + gate * candidate
    }
}

/// Action-conditioned, recurrent next-frame world model.
///
/// A small convolutional encoder observes the current frame. The action row is
/// embedded and injected into a ConvGRU that carries persistent spatial memory.
/// A skip-connected decoder predicts a bounded residual on top of the current
/// frame, which keeps static scenes stable while still allowing motion.
final class WorldModel: Module {
    let configuration: WorldModelConfiguration
    let actionHidden: Linear
    let actionOutput: Linear
    let encoder0: ConvNormActivation
    let encoder1: ConvNormActivation
    let encoder2: ConvNormActivation
    let cell: ConvGRUCell
    let up1: Upsample
    let decoder1: ConvNormActivation
    let up2: Upsample
    let decoder2: ConvNormActivation
    let up3: Upsample
    let decoder3: ConvNormActivation
    let output: Conv2d

    let hiddenHeight: Int
    let hiddenWidth: Int

    init(configuration: WorldModelConfiguration) {
        self.configuration = configuration
        let c1 = configuration.baseChannels
        let c2 = configuration.baseChannels * 2
        let c3 = configuration.recurrentChannels
        let action = configuration.actionEmbeddingChannels
        let h8 = max(1, configuration.height / 8)
        let w8 = max(1, configuration.width / 8)
        hiddenHeight = h8
        hiddenWidth = w8

        actionHidden = Linear(ActionLayout.count, 96)
        actionOutput = Linear(96, action)
        encoder0 = ConvNormActivation(inputChannels: 3, outputChannels: c1, stride: 2)
        encoder1 = ConvNormActivation(inputChannels: c1, outputChannels: c2, stride: 2)
        encoder2 = ConvNormActivation(inputChannels: c2, outputChannels: c3, stride: 2)
        cell = ConvGRUCell(inputChannels: c3 + action, hiddenChannels: c3)
        up1 = Upsample(scaleFactor: 2.0, mode: .nearest)
        decoder1 = ConvNormActivation(inputChannels: c3 + c2 + action, outputChannels: c2)
        up2 = Upsample(scaleFactor: 2.0, mode: .nearest)
        decoder2 = ConvNormActivation(inputChannels: c2 + c1 + action, outputChannels: c1)
        up3 = Upsample(scaleFactor: 2.0, mode: .nearest)
        decoder3 = ConvNormActivation(inputChannels: c1 + 3 + action, outputChannels: c1)
        output = Conv2d(inputChannels: c1, outputChannels: 3, kernelSize: IntOrPair(3), padding: IntOrPair(1))
    }

    func emptyHidden(batchSize: Int) -> [MLXArray] {
        [MLXArray.zeros([batchSize, hiddenHeight, hiddenWidth, configuration.recurrentChannels])]
    }

    private func encode(_ image: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let e0 = encoder0(image)
        let e1 = encoder1(e0)
        let e2 = encoder2(e1)
        return (e0, e1, e2)
    }

    private func actionFeatures(_ actions: MLXArray) -> MLXArray {
        actionOutput(silu(actionHidden(actions)))
    }

    /// Single inference step. `image` is `[B,H,W,3]` in `[-1, 1]`.
    func step(image: MLXArray, action: MLXArray, hidden: [MLXArray]) -> (MLXArray, [MLXArray]) {
        let batch = image.dim(0)
        let (e0, e1, e2) = encode(image)
        let feature = actionFeatures(action)
        let hiddenState = hidden.first ?? emptyHidden(batchSize: batch)[0]
        let h8 = e2.dim(1)
        let w8 = e2.dim(2)
        let actionMap = broadcast(feature.reshaped([batch, 1, 1, feature.dim(-1)]), to: [batch, h8, w8, feature.dim(-1)])
        let updated = cell(concatenated([e2, actionMap], axis: -1), hidden: hiddenState)

        let up1Map = broadcast(feature.reshaped([batch, 1, 1, feature.dim(-1)]),
                               to: [batch, updated.dim(1) * 2, updated.dim(2) * 2, feature.dim(-1)])
        let d1 = decoder1(concatenated([up1(updated), e1, up1Map], axis: -1))
        let up2Map = broadcast(feature.reshaped([batch, 1, 1, feature.dim(-1)]),
                               to: [batch, d1.dim(1) * 2, d1.dim(2) * 2, feature.dim(-1)])
        let d2 = decoder2(concatenated([up2(d1), e0, up2Map], axis: -1))
        let up3Map = broadcast(feature.reshaped([batch, 1, 1, feature.dim(-1)]),
                               to: [batch, d2.dim(1) * 2, d2.dim(2) * 2, feature.dim(-1)])
        let d3 = decoder3(concatenated([up3(d2), image, up3Map], axis: -1))
        let delta = tanh(output(d3))
        let predicted = clip(image + delta, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
        return (predicted, [updated])
    }

    /// Teacher-forced sequence forward used by training.
    /// `frames` is `[B,T+1,H,W,3]`, `actions` is `[B,T,146]`.
    func callAsFunction(frames: MLXArray, actions: MLXArray) -> MLXArray {
        let batch = frames.dim(0)
        let length = actions.dim(1)
        var hidden = emptyHidden(batchSize: batch)
        var predictions: [MLXArray] = []
        predictions.reserveCapacity(length)
        for t in 0..<length {
            let image = frames[0..., t, 0..., 0..., 0...]
            let action = actions[0..., t, 0...]
            let (predicted, next) = step(image: image, action: action, hidden: hidden)
            hidden = next
            predictions.append(predicted)
        }
        return stacked(predictions, axis: 1)
    }

    /// Autoregressive rollout from a seed frame. Returns every generated frame
    /// and the final hidden state.
    func generate(seed: MLXArray, actions: MLXArray, hidden: [MLXArray]) -> (MLXArray, [MLXArray]) {
        let length = actions.dim(1)
        var state = hidden
        var frames: [MLXArray] = []
        var current = seed
        for t in 0..<length {
            let (predicted, next) = step(image: current, action: actions[0..., t, 0...], hidden: state)
            state = next
            current = predicted
            frames.append(predicted)
        }
        return (stacked(frames, axis: 1), state)
    }
}

enum WorldModelLoss {
    static func loss(model: WorldModel, frames: MLXArray, actions: MLXArray, configuration: WorldModelConfiguration) -> MLXArray {
        let predicted = model(frames: frames, actions: actions)
        let target = frames[0..., 1..., 0..., 0..., 0...]
        let pixel = mean(abs(predicted - target))
        var total = pixel
        if configuration.temporalConsistencyWeight > 0, predicted.dim(1) > 1 {
            let predictedDelta = predicted[0..., 1..., 0..., 0..., 0...] - predicted[0..., 0..<(predicted.dim(1) - 1), 0..., 0..., 0...]
            let targetDelta = target[0..., 1..., 0..., 0..., 0...] - target[0..., 0..<(target.dim(1) - 1), 0..., 0..., 0...]
            total = total + Float(configuration.temporalConsistencyWeight) * mean(abs(predictedDelta - targetDelta))
        }
        if configuration.edgeLossWeight > 0 {
            total = total + Float(configuration.edgeLossWeight) * edgeLoss(predicted, target)
        }
        return total
    }

    private static func edgeLoss(_ predicted: MLXArray, _ target: MLXArray) -> MLXArray {
        func gradients(_ value: MLXArray) -> (MLXArray, MLXArray) {
            let dx = value[0..., 0..., 0..., 1..., 0...] - value[0..., 0..., 0..., 0..<(value.dim(3) - 1), 0...]
            let dy = value[0..., 0..., 1..., 0..., 0...] - value[0..., 0..., 0..<(value.dim(2) - 1), 0..., 0...]
            return (dx, dy)
        }
        let (pdx, pdy) = gradients(predicted)
        let (tdx, tdy) = gradients(target)
        return mean(abs(pdx - tdx)) + mean(abs(pdy - tdy))
    }
}
