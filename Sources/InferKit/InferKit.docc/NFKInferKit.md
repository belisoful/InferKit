# ``InferKit/NFKInferKit``

## Overview

`NFKInferKit` reports two versions. ``NFKInferKit/releaseVersion`` is the release, such as `0.4.0`.
``NFKInferKit/APIVersion`` is the binary interface, `"major.minor"`, and several releases can share
one.

The API version covers the Objective-C interface and the C ABI: the classes, selectors, protocols,
constants, enumeration values, functions, and structures the public headers declare, in the core and
in the `@objc` interface InferKitMLX generates. The Swift interface of InferKitMLX is outside it,
because its public Swift API exposes mlx-swift types that change between mlx-swift releases.

- The major version changes only on an incompatible change: a declaration removed or renamed, a type,
  selector, constant value, enumeration value, or structure layout changed, or a required method added
  to a protocol. The minor version then resets to 0.
- The minor version counts additions: a class, method, property, constant, function, enumeration
  case, protocol, or optional protocol method.

The `NFK_API_VERSION_MAJOR` and `NFK_API_VERSION_MINOR` macros record the headers a client compiled
against. A client runs on a library with the same major version and a minor version at least as high,
which an app that loads InferKit from a folder at run time checks before it uses the library:

```objc
BOOL supported = NFKInferKit.APIMajorVersion == NFK_API_VERSION_MAJOR
              && NFKInferKit.APIMinorVersion >= NFK_API_VERSION_MINOR;
```

The API version starts at 1.0. Before the first tagged release it stays 1.0 while the interface
changes, and the first tagged release fixes what 1.0 names. `Docs/installation.md` describes the
release folders an app installs side by side and how it picks the newest one it supports.

## Topics

### Versions

- ``NFKInferKit/releaseVersion``
- ``NFKInferKit/APIVersion``
- ``NFKInferKit/APIMajorVersion``
- ``NFKInferKit/APIMinorVersion``
