//
//  NFKMLXDeepSeekImageProcessor.swift
//  InferKitMLX
//
//  Turning a picture into the patches DeepSeek V4.1's tower reads and the span its decoder holds,
//  from the release's own `inference/image_processor.py`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// The resize plan a picture of a given size follows, and the span it occupies in the prompt.
///
/// @discussion Every later shape follows from this, so it is arithmetic rather than a policy: the
/// picture is padded up to a whole number of patches, shrunk if its token grid would cost more than
/// the release allows, and the aligner's 3x3 downsample decides how many positions the decoder sees.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDeepSeekImagePlan: Sendable, Equatable {
    /// The pixel size the picture is resized and padded to, a whole number of patches on each side.
    public let pixelWidth: Int
    public let pixelHeight: Int
    /// The patch grid the tower reads.
    public let patchRows: Int
    public let patchColumns: Int
    /// The token grid the aligner produces.
    public let tokenRows: Int
    public let tokenColumns: Int

    /// How many positions the span occupies: an opening delimiter, each row of tokens followed by a
    /// newline, and a closing delimiter.
    public var tokenCount: Int { tokenRows * (tokenColumns + 1) + 2 }

    /// How many patches the tower reads.
    public var patchCount: Int { patchRows * patchColumns }
}

/// Turns a picture into the patches DeepSeek V4.1's tower reads.
///
/// @discussion The release's preprocessing is PIL's, and matching it is the whole difficulty. Two
/// details decide whether the patches agree with the reference's, and neither is visible in the
/// Python:
///
/// - PIL resamples an 8-bit image in two passes and rounds to 8 bits BETWEEN them, in fixed point
///   at 22 fractional bits. A float resample of the same coefficients is a different answer by
///   about one part in 255, which is far outside what this package calls parity.
/// - `ImageOps.contain` rounds the aspect-preserving side with Python's `round`, which is
///   round-half-to-even. Swift's `rounded()` is half-away-from-zero, and the two disagree on about
///   one size in eighty.
///
/// The resample runs on the CPU in integer arithmetic for the first reason: it is exact there, it is
/// done once per picture, and float64 on the GPU aborts MLX.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDeepSeekImageProcessor: Sendable {

    public var patchSize: Int
    public var downsampleRatio: Int
    /// The most positions one picture may occupy in the prompt.
    public var maximumTokenCount: Int
    /// The fewest pixels a picture is scaled UP to before planning.
    public var minimumPixels: Int
    /// Where a release sets one, a picture wider than this many times its height is narrowed to the
    /// ratio first, and then resized rather than padded.
    public var maximumWidthHeightRatio: Int?

    public init(patchSize: Int = 14, downsampleRatio: Int = 3, maximumTokenCount: Int = 1024,
                minimumPixels: Int = 544 * 544, maximumWidthHeightRatio: Int? = nil) {
        self.patchSize = patchSize
        self.downsampleRatio = downsampleRatio
        self.maximumTokenCount = maximumTokenCount
        self.minimumPixels = minimumPixels
        self.maximumWidthHeightRatio = maximumWidthHeightRatio
    }

    /// The processor a release's vision configuration describes.
    public init(_ vision: NFKMLXDeepSeekVisionConfiguration) {
        self.init(patchSize: vision.patchSize, downsampleRatio: vision.downsampleRatio,
                  maximumTokenCount: vision.maximumTokenCount,
                  minimumPixels: vision.minimumPixels,
                  maximumWidthHeightRatio: vision.maximumWidthHeightRatio)
    }

    // MARK: - Planning

    /// The grid a picture of this size resizes to.
    public func plan(width sourceWidth: Int, height sourceHeight: Int) -> NFKMLXDeepSeekImagePlan {
        var width = sourceWidth
        var height = sourceHeight
        if let ratio = maximumWidthHeightRatio, width > height * ratio {
            width = height * ratio
        }
        if width * height > 0 && width * height < minimumPixels {
            let scale = (Double(minimumPixels) / Double(width * height)).squareRoot()
            width = Int(Double(width) * scale)
            height = Int(Double(height) * scale)
        }
        var pixelWidth = ceilDivide(width, patchSize) * patchSize
        var pixelHeight = ceilDivide(height, patchSize) * patchSize
        var grid = tokenGrid(pixelHeight: pixelHeight, pixelWidth: pixelWidth)
        if grid.rows * (grid.columns + 1) + 2 > maximumTokenCount {
            (pixelHeight, pixelWidth) = solvedSize(height: height, width: width)
            grid = tokenGrid(pixelHeight: pixelHeight, pixelWidth: pixelWidth)
        }
        return NFKMLXDeepSeekImagePlan(
            pixelWidth: pixelWidth, pixelHeight: pixelHeight,
            patchRows: pixelHeight / patchSize, patchColumns: pixelWidth / patchSize,
            tokenRows: grid.rows, tokenColumns: grid.columns)
    }

    /// The token grid the aligner produces from a pixel size.
    private func tokenGrid(pixelHeight: Int, pixelWidth: Int) -> (rows: Int, columns: Int) {
        (ceilDivide(pixelHeight / patchSize, downsampleRatio),
         ceilDivide(pixelWidth / patchSize, downsampleRatio))
    }

    /// The largest aspect-preserving pixel size whose token grid still fits the budget.
    private func solvedSize(height: Int, width: Int) -> (height: Int, width: Int) {
        let ratio = Double(height) / Double(width)
        let widest = (Double(maximumTokenCount - 2) / ratio + 0.25).squareRoot() - 0.5
        let tallest = widest * ratio
        let cell = patchSize * downsampleRatio
        if widest < 1 { return ((maximumTokenCount - 2) / 2 * cell, cell) }
        if tallest < 1 { return (cell, (maximumTokenCount - 3) * cell) }
        let beta = min(widest.rounded(.down) * Double(cell) / Double(width),
                       tallest.rounded(.down) * Double(cell) / Double(height))
        return (Int(Double(height) * beta / Double(patchSize)) * patchSize,
                Int(Double(width) * beta / Double(patchSize)) * patchSize)
    }

    private func ceilDivide(_ value: Int, _ divisor: Int) -> Int {
        (value + divisor - 1) / divisor
    }

    // MARK: - The span

    /// What a position inside an image span holds, in the order the decoder sees them.
    public func slots(for plan: NFKMLXDeepSeekImagePlan, startingAt start: Int)
        -> [(position: Int, slot: NFKMLXDeepSeekImageDelimiters.Slot)] {
        var slots: [(position: Int, slot: NFKMLXDeepSeekImageDelimiters.Slot)] = [(start, .start)]
        var row = 0
        for line in 0 ..< plan.tokenRows {
            for column in 0 ..< plan.tokenColumns {
                slots.append((start + slots.count, .token(line * plan.tokenColumns + column)))
                row += 1
            }
            slots.append((start + slots.count, .newline))
        }
        slots.append((start + slots.count, .end))
        return slots
    }

    // MARK: - Pixels to patches

    /// The patches the tower reads, from an RGB picture.
    ///
    /// - Parameter pixels: `[height, width, 3]`, each channel 0 through 255.
    /// - Parameter width: the picture's width in pixels.
    /// - Parameter height: the picture's height in pixels.
    /// - Parameter plan: the resize and tiling chosen for the picture.
    /// - Returns: `[patchCount, 3, patchSize, patchSize]` in reading order, normalized to -1...1.
    public func patches(from pixels: [UInt8], width: Int, height: Int,
                        plan: NFKMLXDeepSeekImagePlan) -> MLXArray {
        let resized = resized(pixels, width: width, height: height, plan: plan)
        // `(x / 255 - 0.5) / 0.5`, then rounded to bfloat16 because the reference casts there
        // before it cuts the patches. Skipping the cast leaves a difference around 4e-3, which is
        // the format's own step and not a mistake anywhere else.
        let scaled = MLXArray(resized.map(Float.init))
            .reshaped([plan.pixelHeight, plan.pixelWidth, 3]) / 255
        let normalized = ((scaled - 0.5) / 0.5).asType(.bfloat16).asType(.float32)
        let planar = normalized.transposed(2, 0, 1)                 // [3, height, width]
        return planar
            .reshaped([3, plan.patchRows, patchSize, plan.patchColumns, patchSize])
            .transposed(1, 3, 0, 2, 4)
            .reshaped([plan.patchCount, 3, patchSize, patchSize])
    }

    /// The picture at the plan's pixel size: shrunk to fit and padded with grey, or resized outright
    /// where a release's width-to-height ratio put it on that branch.
    ///
    /// @discussion `ImageOps.pad` contains the picture inside the target and centers what is left
    /// over, padding with (127, 127, 127). Only one axis is ever padded, because `contain` matches
    /// the other exactly.
    func resized(_ pixels: [UInt8], width: Int, height: Int,
                 plan: NFKMLXDeepSeekImagePlan) -> [UInt8] {
        if let ratio = maximumWidthHeightRatio, width >= ratio * height {
            return resampled(pixels, width: width, height: height,
                             toWidth: plan.pixelWidth, toHeight: plan.pixelHeight)
        }
        let (fitWidth, fitHeight) = contained(width: width, height: height,
                                              inWidth: plan.pixelWidth, inHeight: plan.pixelHeight)
        let fitted = resampled(pixels, width: width, height: height,
                               toWidth: fitWidth, toHeight: fitHeight)
        if fitWidth == plan.pixelWidth && fitHeight == plan.pixelHeight { return fitted }

        var padded = [UInt8](repeating: 127, count: plan.pixelWidth * plan.pixelHeight * 3)
        let left = fitWidth != plan.pixelWidth ? (plan.pixelWidth - fitWidth) / 2 : 0
        let top = fitWidth != plan.pixelWidth ? 0 : (plan.pixelHeight - fitHeight) / 2
        for row in 0 ..< fitHeight {
            let source = row * fitWidth * 3
            let destination = ((row + top) * plan.pixelWidth + left) * 3
            padded.replaceSubrange(destination ..< (destination + fitWidth * 3),
                                   with: fitted[source ..< (source + fitWidth * 3)])
        }
        return padded
    }

    /// The size `ImageOps.contain` shrinks a picture to: the target on one axis, the aspect-
    /// preserving size on the other.
    ///
    /// The rounding is `.toNearestOrEven` because Python's `round` is, and about one size in eighty
    /// lands exactly on a half where the two disagree.
    func contained(width: Int, height: Int, inWidth: Int, inHeight: Int) -> (width: Int, height: Int) {
        let source = Double(width) / Double(height)
        let target = Double(inWidth) / Double(inHeight)
        if source == target { return (inWidth, inHeight) }
        if source > target {
            let scaled = (Double(height) / Double(width) * Double(inWidth)).rounded(.toNearestOrEven)
            return (inWidth, Int(scaled))
        }
        let scaled = (Double(width) / Double(height) * Double(inHeight)).rounded(.toNearestOrEven)
        return (Int(scaled), inHeight)
    }

    // MARK: - PIL's resample, exactly

    /// The fractional bits PIL carries its 8-bit resample in.
    private static let precisionBits = 32 - 8 - 2

    /// Resamples an 8-bit RGB picture the way PIL does: horizontally into an 8-bit intermediate,
    /// then vertically, each pass accumulating in fixed point and rounding once at the end.
    private func resampled(_ pixels: [UInt8], width: Int, height: Int,
                           toWidth: Int, toHeight: Int) -> [UInt8] {
        var image = pixels
        if toWidth != width {
            image = pass(image, rows: height, columns: width, toColumns: toWidth)
        }
        guard toHeight != height else { return image }
        // The vertical pass is the horizontal one over a transposed picture, which keeps one
        // implementation of the arithmetic rather than two that could drift apart.
        let columns = toWidth
        var transposed = [UInt8](repeating: 0, count: columns * height * 3)
        for row in 0 ..< height {
            for column in 0 ..< columns {
                let from = (row * columns + column) * 3
                let to = (column * height + row) * 3
                transposed[to] = image[from]
                transposed[to + 1] = image[from + 1]
                transposed[to + 2] = image[from + 2]
            }
        }
        let scaled = pass(transposed, rows: columns, columns: height, toColumns: toHeight)
        var result = [UInt8](repeating: 0, count: columns * toHeight * 3)
        for column in 0 ..< columns {
            for row in 0 ..< toHeight {
                let from = (column * toHeight + row) * 3
                let to = (row * columns + column) * 3
                result[to] = scaled[from]
                result[to + 1] = scaled[from + 1]
                result[to + 2] = scaled[from + 2]
            }
        }
        return result
    }

    /// One horizontal resampling pass over `[rows, columns, 3]` bytes.
    private func pass(_ pixels: [UInt8], rows: Int, columns: Int, toColumns: Int) -> [UInt8] {
        let weights = Self.coefficients(from: columns, to: toColumns)
        let half = 1 << (Self.precisionBits - 1)
        var result = [UInt8](repeating: 0, count: rows * toColumns * 3)
        for row in 0 ..< rows {
            for column in 0 ..< toColumns {
                let span = weights[column]
                var sums = (half, half, half)
                for (offset, weight) in span.weights.enumerated() {
                    let source = (row * columns + span.start + offset) * 3
                    sums.0 += Int(pixels[source]) * weight
                    sums.1 += Int(pixels[source + 1]) * weight
                    sums.2 += Int(pixels[source + 2]) * weight
                }
                let destination = (row * toColumns + column) * 3
                result[destination] = Self.clipped(sums.0)
                result[destination + 1] = Self.clipped(sums.1)
                result[destination + 2] = Self.clipped(sums.2)
            }
        }
        return result
    }

    private static func clipped(_ accumulated: Int) -> UInt8 {
        let value = accumulated >> precisionBits
        return UInt8(Swift.max(0, Swift.min(255, value)))
    }

    /// PIL's bicubic coefficients for one axis, quantized to the fixed point its 8-bit path uses.
    ///
    /// @discussion The cubic coefficient is -0.5, which is PIL's and NOT the -0.75 that an
    /// un-antialiased torch bicubic uses; `NFKMLXRFDetr` records the same constant as the one seam
    /// bug its released-weight parity caught. The support widens with the scale when shrinking,
    /// which is what antialiasing means here.
    static func coefficients(from inSize: Int, to outSize: Int)
        -> [(start: Int, weights: [Int])] {
        func filter(_ value: Double) -> Double {
            let a = -0.5
            let x = abs(value)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        let scale = Double(inSize) / Double(outSize)
        let support = scale > 1 ? 2 * scale : 2
        let inverse = scale > 1 ? 1 / scale : 1
        return (0 ..< outSize).map { index in
            let center = scale * (Double(index) + 0.5)
            let start = Swift.max(Int(center - support + 0.5), 0)
            let end = Swift.min(Int(center + support + 0.5), inSize)
            var raw = [Double]()
            var total: Double = 0
            for source in start ..< end {
                let weight = filter((Double(source) - center + 0.5) * inverse)
                raw.append(weight)
                total += weight
            }
            let quantized = raw.map { weight -> Int in
                let normalized = total != 0 ? weight / total : weight
                let scaled = normalized * Double(1 << precisionBits)
                return Int(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            return (start, quantized)
        }
    }
}


/// The image stack a release carries beside its decoder.
///
/// @discussion Four pieces turn a picture into positions the decoder reads: the preprocessor that
/// plans the grid and cuts the patches, the tower that reads them, the aligner that pools them into
/// the decoder's width, and the learned delimiters that open, close and line-break the span. They
/// are grouped because no one of them is useful alone and because a release stores them together.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXDeepSeekImageStack {
    public let tower: NFKMLXDeepSeekVisionNet
    public let aligner: NFKMLXDeepSeekAligner
    public let delimiters: NFKMLXDeepSeekImageDelimiters
    public let processor: NFKMLXDeepSeekImageProcessor
    public let configuration: NFKMLXDeepSeekVisionConfiguration

    public init(_ vision: NFKMLXDeepSeekVisionConfiguration) {
        configuration = vision
        tower = NFKMLXDeepSeekVisionNet(vision)
        aligner = NFKMLXDeepSeekAligner(vision)
        delimiters = NFKMLXDeepSeekImageDelimiters(outputSize: vision.outputSize)
        processor = NFKMLXDeepSeekImageProcessor(vision)
    }

    /// One picture, ready to be written into a prompt.
    public struct Span {
        /// `[tokenRows * tokenColumns, hidden]`, the aligner's pooled tokens in reading order.
        public let aligned: MLXArray
        /// Where each position of the span sits and what it holds.
        public let slots: [(position: Int, slot: NFKMLXDeepSeekImageDelimiters.Slot)]
        public let plan: NFKMLXDeepSeekImagePlan
    }

    /// The decoder-width tokens a picture becomes.
    public func span(forPixels pixels: [UInt8], width: Int, height: Int,
                     startingAt start: Int) -> Span {
        let plan = processor.plan(width: width, height: height)
        let patches = processor.patches(from: pixels, width: width, height: height, plan: plan)
        let features = tower(patches, rows: plan.patchRows, columns: plan.patchColumns)
        let aligned = aligner(features, rows: plan.patchRows, columns: plan.patchColumns)
        return Span(aligned: aligned, slots: processor.slots(for: plan, startingAt: start),
                    plan: plan)
    }

    /// Expands each image placeholder in `tokens` into the span its picture occupies.
    ///
    /// @discussion Every position of a span carries the placeholder id, exactly as the release
    /// writes it: only the span's layout tells the positions apart, which is why the decoder needs
    /// the mask as well as the embeddings.
    public func expanded(_ tokens: [Int], pictures: [(pixels: [UInt8], width: Int, height: Int)])
        throws -> (tokens: [Int], spans: [Span]) {
        let placeholder = configuration.imageTokenID
        let found = tokens.filter { $0 == placeholder }.count
        guard found == pictures.count else {
            throw NFKMLXError.unsupportedConfiguration(
                "the prompt holds \(found) image placeholders (token \(placeholder)) and "
                + "\(pictures.count) pictures were supplied")
        }
        var expanded = [Int]()
        var spans = [Span]()
        var next = pictures.makeIterator()
        for token in tokens {
            guard token == placeholder, let picture = next.next() else {
                expanded.append(token)
                continue
            }
            let made = span(forPixels: picture.pixels, width: picture.width,
                            height: picture.height, startingAt: expanded.count)
            spans.append(made)
            expanded.append(contentsOf: repeatElement(placeholder, count: made.plan.tokenCount))
        }
        return (expanded, spans)
    }

    /// The embeddings and the span mask a prompt with pictures runs with.
    ///
    /// - Parameter tokens: the prompt's ids, with each picture's placeholder run expanded.
    /// - Parameter spans: the pictures' spans, in the order they appear in `tokens`.
    /// - Parameter embed: the decoder's own token embedding, which the text positions keep.
    public func inputs(for tokens: [Int], spans: [Span],
                       embed: (MLXArray) -> MLXArray) -> (embeddings: MLXArray, images: MLXArray) {
        var embeddings = embed(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))
        var marked = [Bool](repeating: false, count: tokens.count)
        for span in spans {
            embeddings = delimiters.merged(into: embeddings, aligned: span.aligned,
                                           slots: span.slots)
            for (position, _) in span.slots where position < marked.count { marked[position] = true }
        }
        return (embeddings, MLXArray(marked).reshaped([1, tokens.count]))
    }
}
