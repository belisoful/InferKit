Pod::Spec.new do |s|
  s.name             = 'InferKit'
  s.version          = '0.1.0'
  s.summary          = 'A small, cross-platform inference toolkit for Objective-C.'
  s.description      = <<-DESC
    InferKit is an Objective-C inference toolkit: a swappable backend protocol, request/result
    value types, a thread-safe async job handle, and shipped backends (passthrough mock,
    in-process Core ML, an OpenAI-compatible remote client, and a submit-poll-fetch generation
    base). It adds an RGBA-interleaved to planar CHW/HWC tensor conversion, an MLMultiArray
    bridge, and a Hugging Face model-download layer. It has no FxPlug or host-framework
    dependency, so any Metal/Apple app can use it. A consumer brings a heavier runtime (MLX, a
    C or Rust engine) by adopting the backend protocol.
  DESC
  s.homepage         = 'https://github.com/belisoful/InferKit'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Brad Anderson' => 'belisoful@icloud.com' }

  # CocoaPods fetches the pod at this git tag; the tag must exist before release.
  s.source           = { :git => 'https://github.com/belisoful/InferKit.git', :tag => "v#{s.version}" }

  s.osx.deployment_target  = '11.0'
  s.ios.deployment_target  = '14.0'
  s.tvos.deployment_target = '14.0'

  # Source pod: the same files SwiftPM (Package.swift) compiles. The MLX companion
  # (InferKitMLX) is SwiftPM-only, so it is deliberately excluded here.
  s.source_files        = 'Sources/InferKit/**/*.{h,m}'
  s.public_header_files  = 'Sources/InferKit/include/InferKit/*.h'
  # Sibling headers directly under Sources/InferKit are private; the public API is include/InferKit.
  s.private_header_files = 'Sources/InferKit/*.h'
  s.requires_arc        = true

  # System frameworks the sources link against.
  s.frameworks = 'Foundation', 'CoreML', 'CoreVideo', 'CoreGraphics', 'Metal', 'IOSurface'
end
