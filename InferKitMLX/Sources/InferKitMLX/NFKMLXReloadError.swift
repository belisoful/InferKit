//
//  NFKMLXReloadError.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation

/// Error codes in `NFKMLXReloadErrorDomain`: why a backend cannot load its model again after an unload.
///
/// @discussion A backend that unloads loads its model again on the next run or `prepare`, from the
/// local files it was built from. That load reads the disk only and makes no network request.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXReloadError)
public enum NFKMLXReloadError: Int, Error, CustomNSError {

    /// The files the model was built from are no longer on disk, such as after a removal through
    /// `NFKHFHub`. The error's `NSFilePathErrorKey` names the first missing file. Downloading the files
    /// again through an `NFKDownload` puts them back where the backend reads them, and its next run
    /// loads them. Introduced in InferKit 0.4.0.
    case modelFilesRemoved = 1

    /// `NFKMLXReloadErrorDomain`. Introduced in InferKit 0.4.0.
    public static var errorDomain: String { "InferKitMLX.NFKMLXReloadError" }
}
