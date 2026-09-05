//
//  NFKMLXReferenceModels.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

/// Reference MLX models registered with `NFKMLXModelRegistry`, so an Objective-C consumer can build
/// and run a local MLX model by name. These are the pattern a learned model follows: a keyer such as
/// CorridorKey's GreenFormer registers the same way, loading its weights from the URL and running its
/// network in the forward.
@objc(NFKMLXReferenceModels)
public final class NFKMLXReferenceModels: NSObject {

    /// Registers every model InferKitMLX ships in one call: the reference stand-ins (`green-screen-keyer`,
    /// `tone-speech`, and the `diffusion-*` oracle pipelines) and the real models (`real-esrgan-x4`,
    /// `depth-anything-v2-small`, `lama-inpaint`, `sd-inpaint`). Call once at launch, then build any of
    /// them by name through `NFKMLXModelRegistry`.
    @objc public static func registerAll() {
        registerGreenScreenKeyer()
        registerToneSpeech()
        NFKMLXWhisper.register()
        NFKMLXDemucs.register()
        NFKMLXHTDemucs.register()
        registerDiffusionUpscaler()
        registerDiffusionDepth()
        registerDiffusionInpainter()
        registerControlNet()
        NFKMLXRealESRGAN.register()
        NFKMLXDepthAnything.register()
        NFKMLXDepthAnything3.register()
        NFKMLXU2Net.register()
        NFKMLXNAFNet.register()
        NFKMLXSAM.register()
        NFKMLXRIFE.register()
        NFKMLXRIFEv4.register()
        NFKMLXRAFT.register()
        NFKMLXLaMa.register()
        NFKMLXStableDiffusionInpaint.register()
        NFKMLXMarigold.register()
        NFKMLXSDUpscaler.register()
        NFKMLXStyleTransfer.register()
        NFKMLXCLIP.register()
        NFKMLXRVM.register()
        NFKMLXCodeFormer.register()
        NFKMLXZeroDCE.register()
        NFKMLXMODNet.register()
        NFKMLXYOLO.register()
        NFKMLXRTDetr.register()
        NFKMLXRFDetr.register()
        NFKMLXRetinaFace.register()
        NFKMLXSegFormer.register()
        NFKMLXSwinIR.register()
        NFKMLXColorizer.register()
        NFKMLXSiggraphColorizer.register()
        NFKMLXPose.register()
        NFKMLXDeepLab.register()
        NFKMLXConvTasNet.register()
        NFKMLXDenoiser.register()
        NFKMLXVAD.register()
        NFKMLXSileroVAD.register()
        NFKMLXDAC.register()
        NFKMLXSNAC.register()
        NFKMLXSigLIP2.register()
        NFKMLXTAESD.register()
        NFKMLXAudioTagger.register()
        NFKMLXBiSeNet.register()
        NFKMLXBiSeNetV2.register()
        NFKMLXVideoSR.register()
        NFKMLXMusic3.register()
    }

    /// Registers a simple text-to-speech synthesizer under `"tone-speech"`. The synth turns each
    /// character into a short tone whose pitch follows the character, a working stand-in for a learned
    /// TTS model and a reference for wiring `NFKMLXSpeechBackend` (text in, an audio file out).
    @objc public static func registerToneSpeech() {
        NFKMLXModelRegistry.register(name: "tone-speech") { _ in
            NFKMLXSpeechBackend(identifier: "tone-speech") { text, sampleRate in
                let samplesPerCharacter = sampleRate / 10          // 0.1 s per character
                var samples = [Float]()
                samples.reserveCapacity(max(text.unicodeScalars.count, 1) * samplesPerCharacter)
                for scalar in text.unicodeScalars {
                    let frequency = 120.0 + Double(scalar.value % 800)
                    for n in 0 ..< samplesPerCharacter {
                        let time = Double(n) / Double(sampleRate)
                        samples.append(Float(sin(2 * .pi * frequency * time)) * 0.3)
                    }
                }
                if samples.isEmpty { samples = [0] }
                return samples.withUnsafeBufferPointer { MLXArray($0, [samples.count]) }
            }
        }
    }

    /// Registers a simple green-screen keyer under `"green-screen-keyer"`. The forward keeps the plate
    /// as the straight foreground and derives an alpha matte from how strongly green dominates each
    /// pixel — a working stand-in for a learned keyer, and a reference for the registration pattern.
    @objc public static func registerGreenScreenKeyer() {
        NFKMLXModelRegistry.register(name: "green-screen-keyer") { _ in
            var configuration = NFKMattingConfiguration()
            configuration.emitsMatte = true
            return NFKMLXMattingBackend(identifier: "green-screen-keyer", configuration: configuration) { plate, _ in
                let channels = split(plate, parts: 3, axis: 2)          // [R, G, B], each [H, W, 1]
                let greenness = channels[1] - maximum(channels[0], channels[2])
                let alpha = clip(1.0 - greenness * 4.0, min: 0.0, max: 1.0)   // strong green -> low alpha
                return concatenated([plate, alpha], axis: 2)            // [H, W, 4]: straight fg + matte
            }
        }
    }
}
