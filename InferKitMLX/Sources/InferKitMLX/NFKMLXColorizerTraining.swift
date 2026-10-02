//
//  NFKMLXColorizerTraining.swift
//  InferKitMLX
//
//  The ECCV-16 colorizer's fine-tune, ported from richzhang/colorization's `caffe` branch at a1642d6:
//  `colorization_train_val_v2.prototxt`, the Python layers of `resources/caffe_traininglayers.py` (the soft
//  encoding, the gray mask, and the prior boost), the `SoftmaxCrossEntropyLoss` layer, and `solver.prototxt`.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// sRGB to CIELAB as both colorizers' training data converts it: scikit-image's `rgb2lab` constants, which
/// richzhang/colorization's `BGR2LabLayer` calls and colorization-pytorch's `util.rgb2lab` writes out.
enum NFKColorizerTrainingLab {
    /// `[N, H, W, 3]` sRGB in [0, 1] → L in 0…100 and ab, `[N, H, W, 3]`.
    static func reference(_ rgb: MLXArray) -> MLXArray {
        let linear = MLX.where(rgb .> 0.04045, pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        let (r, g, b) = (linear[.ellipsis, 0], linear[.ellipsis, 1], linear[.ellipsis, 2])
        let x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.95047
        let y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        let z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883
        func f(_ t: MLXArray) -> MLXArray {
            MLX.where(t .> 0.008856, pow(MLX.maximum(t, 0), 1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
        }
        let (fx, fy, fz) = (f(x), f(y), f(z))
        return stacked([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis: -1)
    }
}

/// The objective the ECCV-16 colorizer trains with: a cross-entropy between the 313 chroma bins' softmax and
/// a soft encoding of the true ab, rebalanced toward rare colors.
///
/// @discussion Each pixel's true ab is encoded over its ten nearest bin centers with Gaussian weights of
/// deviation 5 (`NNEncLayer`). The loss is `SoftmaxCrossEntropyLoss`'s `Σ t·(log t − log p)` over the pixels
/// and bins, divided by the batch size. `ClassRebalanceMultLayer` scales each pixel's gradient, and only its
/// gradient, by the prior factor of its nearest bin (`PriorBoostLayer`, the prior mixed half with uniform),
/// zeroed for an image whose ab never exceeds 5 (`NonGrayMaskLayer`). The returned loss carries that
/// weight, so its gradient is the reference's; loss(logits:ab:binCenters:) also returns the reference's
/// unweighted value.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXColorizerObjective: Sendable {
    public var neighbors = 10
    public var sigma: Float = 5
    /// The uniform share mixed into the prior before inverting it, `PriorBoostLayer`'s `gamma`.
    public var priorMix: Float = 0.5
    /// The ab magnitude below which an image counts as grayscale.
    public var grayThreshold: Float = 5
    /// The bins' frequencies over the training set, `resources/prior_probs.npy` for ImageNet by default.
    public var priorProbabilities: [Float]

    public init(priorProbabilities: [Float] = NFKMLXColorizerObjective.imageNetPrior) {
        self.priorProbabilities = priorProbabilities
    }

    /// Scores the network on normalized lightness `[N, H, W, 1]` against the true ab `[N, H/4, W/4, 2]`, the
    /// pair `NFKMLXColorizer.trainingExample(_:)` returns.
    public func callAsFunction(_ net: NFKMLXColorizerNet, _ lightness: MLXArray, _ ab: MLXArray) -> MLXArray {
        loss(logits: net.logits(lightness), ab: ab, binCenters: Self.binCenters).rebalanced
    }

    /// `resources/pts_in_hull.npy`: the 313 bins' ab centers `[313, 2]`, on a 10-unit grid inside the sRGB
    /// gamut. The released network's annealed-mean readout holds the same centers over 110.
    public static var binCenters: MLXArray { MLXArray(hullPoints.map(Float.init), [313, 2]) }

    static let hullPoints: [Int] = [

        -90, 50, -90, 60, -90, 70, -90, 80, -90, 90, -80, 20, -80, 30, -80, 40,
        -80, 50, -80, 60, -80, 70, -80, 80, -80, 90, -70, 0, -70, 10, -70, 20,
        -70, 30, -70, 40, -70, 50, -70, 60, -70, 70, -70, 80, -70, 90, -60, -20,
        -60, -10, -60, 0, -60, 10, -60, 20, -60, 30, -60, 40, -60, 50, -60, 60,
        -60, 70, -60, 80, -60, 90, -50, -30, -50, -20, -50, -10, -50, 0, -50, 10,
        -50, 20, -50, 30, -50, 40, -50, 50, -50, 60, -50, 70, -50, 80, -50, 90,
        -50, 100, -40, -40, -40, -30, -40, -20, -40, -10, -40, 0, -40, 10, -40, 20,
        -40, 30, -40, 40, -40, 50, -40, 60, -40, 70, -40, 80, -40, 90, -40, 100,
        -30, -50, -30, -40, -30, -30, -30, -20, -30, -10, -30, 0, -30, 10, -30, 20,
        -30, 30, -30, 40, -30, 50, -30, 60, -30, 70, -30, 80, -30, 90, -30, 100,
        -20, -50, -20, -40, -20, -30, -20, -20, -20, -10, -20, 0, -20, 10, -20, 20,
        -20, 30, -20, 40, -20, 50, -20, 60, -20, 70, -20, 80, -20, 90, -20, 100,
        -10, -60, -10, -50, -10, -40, -10, -30, -10, -20, -10, -10, -10, 0, -10, 10,
        -10, 20, -10, 30, -10, 40, -10, 50, -10, 60, -10, 70, -10, 80, -10, 90,
        -10, 100, 0, -70, 0, -60, 0, -50, 0, -40, 0, -30, 0, -20, 0, -10,
        0, 0, 0, 10, 0, 20, 0, 30, 0, 40, 0, 50, 0, 60, 0, 70,
        0, 80, 0, 90, 0, 100, 10, -80, 10, -70, 10, -60, 10, -50, 10, -40,
        10, -30, 10, -20, 10, -10, 10, 0, 10, 10, 10, 20, 10, 30, 10, 40,
        10, 50, 10, 60, 10, 70, 10, 80, 10, 90, 20, -80, 20, -70, 20, -60,
        20, -50, 20, -40, 20, -30, 20, -20, 20, -10, 20, 0, 20, 10, 20, 20,
        20, 30, 20, 40, 20, 50, 20, 60, 20, 70, 20, 80, 20, 90, 30, -90,
        30, -80, 30, -70, 30, -60, 30, -50, 30, -40, 30, -30, 30, -20, 30, -10,
        30, 0, 30, 10, 30, 20, 30, 30, 30, 40, 30, 50, 30, 60, 30, 70,
        30, 80, 30, 90, 40, -100, 40, -90, 40, -80, 40, -70, 40, -60, 40, -50,
        40, -40, 40, -30, 40, -20, 40, -10, 40, 0, 40, 10, 40, 20, 40, 30,
        40, 40, 40, 50, 40, 60, 40, 70, 40, 80, 40, 90, 50, -100, 50, -90,
        50, -80, 50, -70, 50, -60, 50, -50, 50, -40, 50, -30, 50, -20, 50, -10,
        50, 0, 50, 10, 50, 20, 50, 30, 50, 40, 50, 50, 50, 60, 50, 70,
        50, 80, 60, -110, 60, -100, 60, -90, 60, -80, 60, -70, 60, -60, 60, -50,
        60, -40, 60, -30, 60, -20, 60, -10, 60, 0, 60, 10, 60, 20, 60, 30,
        60, 40, 60, 50, 60, 60, 60, 70, 60, 80, 70, -110, 70, -100, 70, -90,
        70, -80, 70, -70, 70, -60, 70, -50, 70, -40, 70, -30, 70, -20, 70, -10,
        70, 0, 70, 10, 70, 20, 70, 30, 70, 40, 70, 50, 70, 60, 70, 70,
        70, 80, 80, -110, 80, -100, 80, -90, 80, -80, 80, -70, 80, -60, 80, -50,
        80, -40, 80, -30, 80, -20, 80, -10, 80, 0, 80, 10, 80, 20, 80, 30,
        80, 40, 80, 50, 80, 60, 80, 70, 90, -110, 90, -100, 90, -90, 90, -80,
        90, -70, 90, -60, 90, -50, 90, -40, 90, -30, 90, -20, 90, -10, 90, 0,
        90, 10, 90, 20, 90, 30, 90, 40, 90, 50, 90, 60, 90, 70, 100, -90,
        100, -80, 100, -70, 100, -60, 100, -50, 100, -40, 100, -30, 100, -20, 100, -10,
        100, 0,
    ]

    /// The rebalanced loss, whose gradient is the reference's, and the reference's unweighted value, over the
    /// bin centers `[bins, 2]` (``binCenters`` for the released network).
    public func loss(logits: MLXArray, ab: MLXArray, binCenters: MLXArray) -> (rebalanced: MLXArray, crossEntropy: MLXArray) {
        let (batch, bins) = (logits.dim(0), logits.dim(-1))
        let target = encoding(ab.reshaped([-1, 2]), binCenters: binCenters).reshaped(logits.shape)
        let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
        let kept = target .> 0
        let terms = MLX.where(kept, target * (log(MLX.where(kept, target, MLXArray(Float(1)))) - logProbabilities),
                              MLXArray(Float(0))).sum(axis: -1)
        let nearest = argMax(target, axis: -1)
        let colored = (abs(ab) .> grayThreshold).asType(.float32).max(axes: [1, 2, 3]).reshaped([batch, 1, 1])
        let weight = stopGradient(priorFactor[nearest] * colored)
        _ = bins
        return ((terms * weight).sum() / Float(batch), terms.sum() / Float(batch))
    }

    /// `NNEncode.encode_points_mtx_nd`: Gaussian weights over each point's nearest bins, normalized.
    func encoding(_ points: MLXArray, binCenters: MLXArray) -> MLXArray {
        let distances = square(points.expandedDimensions(axis: 1) - binCenters.expandedDimensions(axis: 0)).sum(axis: -1)
        let rank = argSort(argSort(distances, axis: -1), axis: -1)
        let weights = MLX.where(rank .< neighbors, exp(-distances / (2 * sigma * sigma)), MLXArray(Float(0)))
        return weights / weights.sum(axis: -1, keepDims: true)
    }

    /// `PriorFactor`: the prior mixed with the uniform over its nonzero bins, inverted, and normalized so its
    /// expectation under the prior is one.
    var priorFactor: MLXArray {
        let prior = priorProbabilities.map(Double.init)
        let support = Double(prior.filter { $0 != 0 }.count)
        let mixed = prior.map { (1 - Double(priorMix)) * $0 + Double(priorMix) * ($0 != 0 ? 1 / support : 0) }
        let inverted = mixed.map { 1 / $0 }
        let expectation = zip(prior, inverted).reduce(0) { $0 + $1.0 * $1.1 }
        return MLXArray(inverted.map { Float($0 / expectation) })
    }

    /// `resources/prior_probs.npy`: the 313 bins' frequencies over ImageNet.
    public static let imageNetPrior: [Float] = [
        4.886370774683732e-08, 1.385178992542849e-07, 5.765753296328652e-07, 2.6393382123317163e-06, 9.503124277991785e-07, 3.566914661791634e-08,
        2.537012182099119e-07, 7.525312855284019e-07, 1.9617667000117203e-06, 3.0148059201290855e-06, 4.886445836965772e-06, 5.640534025064313e-06,
        1.2942927343984957e-06, 9.366798213092383e-08, 6.353914470771186e-07, 2.246414150756667e-06, 5.2555358040485195e-06, 8.7660741422368e-06,
        1.340916545621541e-05, 1.8277922912987378e-05, 1.6072496428774676e-05, 7.2220233499552934e-06, 1.1914289047493368e-06, 4.1888778153205936e-07,
        1.8878667100513214e-06, 5.562571220030682e-06, 1.3943089012476055e-05, 2.407920260236458e-05, 3.4740368255535504e-05, 4.8228069592620715e-05,
        6.844665387516861e-05, 7.227637683936883e-05, 4.296251573846445e-05, 1.6485100269231254e-05, 3.025684343449224e-06, 7.738051659960398e-07,
        1.6446115064538475e-05, 4.387410862504647e-05, 5.87340627985586e-05, 8.723163151668243e-05, 0.00011877063180798355, 0.00016708896103410247,
        0.00026325249386502966, 0.0003385184526332533, 0.0002603470295497782, 0.000126975044583059, 4.089617544913256e-05, 7.228175635278452e-06,
        1.996938728407485e-07, 8.92095221887288e-07, 2.1768106293344453e-05, 0.00011251304912612339, 0.00021637953763431705, 0.00026603057686519146,
        0.0003322761251458244, 0.0005058861309854875, 0.0009307366435187428, 0.0013746057427333322, 0.001223998255449075, 0.0006849419666828104,
        0.00026633175001933925, 7.812691590938548e-05, 1.5124833395246015e-05, 6.919090150025826e-07, 7.844424072129307e-07, 2.6044292350964743e-05,
        0.00017322590351060826, 0.00042618323148242346, 0.0006943514382180301, 0.0009101066074866882, 0.0013639643455049706, 0.002656139566651216,
        0.004285494677153395, 0.004122079516354024, 0.0024615193363116054, 0.0010569683479912694, 0.0003652799814727483, 0.00011669796002511184,
        3.933794649136768e-05, 5.51556882580345e-06, 2.3173647412656074e-05, 0.00021886435960656613, 0.000700249521967109, 0.0013998257110752422,
        0.002467696267256372, 0.004125440035044677, 0.006659526843287239, 0.009481453507707644, 0.009471519614150653, 0.006014234411242494,
        0.00273632660045453, 0.001059316389505778, 0.00042481314778116135, 0.0002066089704890517, 0.00011251665964511477, 1.823437981972153e-05,
        1.298062073769513e-05, 0.0001984596479374361, 0.0010234851992861305, 0.0030212100535917884, 0.006834732373424585, 0.019154820690375076,
        0.04910890701194649, 0.03734361643489669, 0.021541026271199578, 0.01195762487759053, 0.005374960202313204, 0.0022204508311087757,
        0.001002283685140692, 0.0005345221641792241, 0.000298875772328174, 0.00011331255318585796, 7.1443113838299625e-06, 5.098910973863863e-06,
        0.00011701696762946952, 0.0007916087759901634, 0.0028516033125571705, 0.006631060031776227, 0.015282381400006942, 0.06420292217830871,
        0.2012340113556749, 0.10829850923849671, 0.04150763834132455, 0.01655147589345126, 0.00617025337653934, 0.0024903778691890057,
        0.0011721089919714265, 0.000626213708021392, 0.00031961592914015894, 7.251744436501847e-05, 1.5689404911157703e-06, 1.3685372059933345e-06,
        4.8093222371367795e-05, 0.00035480824041937296, 0.001198596734082741, 0.002426558860031101, 0.0037785252625200153, 0.0066116848446421544,
        0.019560146438858814, 0.0531084299775812, 0.048359484234070604, 0.03273124495672014, 0.017650565433516993, 0.007574831620606694,
        0.003076624887064972, 0.001316429431680701, 0.0006089197448305731, 0.00025108509510363413, 2.9883528930830048e-05, 1.6299051135426677e-05,
        0.00015067354955696218, 0.0004487431647653595, 0.0008112590899346631, 0.0010288900843513978, 0.0010708981263452065, 0.001195690575756661,
        0.002191572109778545, 0.005414704062354201, 0.010513671789486022, 0.01224076977500643, 0.00983720009552366, 0.005755964175524455,
        0.002788410705371128, 0.001229070760693792, 0.000518767129195145, 0.00016017383751180684, 8.733085202408846e-06, 3.67741205092613e-06,
        6.018622898493446e-05, 0.00019501417189004243, 0.0003290275113810248, 0.0003921089615045731, 0.0003364672023114458, 0.0002875807372741427,
        0.00035586498154449185, 0.0006000250383578096, 0.0013030295669700122, 0.0029426484776852316, 0.004355100751640371, 0.004232531059674437,
        0.0031058469788196354, 0.0018508245974295495, 0.0009275907439405732, 0.0003965217797114613, 8.902972989720967e-05, 2.16968518920351e-06,
        5.712861746237908e-07, 2.036258350079038e-05, 9.382339655005752e-05, 0.0001534005721271283, 0.0001732895819447461, 0.00013987396172288917,
        0.00010541354920365735, 0.00012405359996794373, 0.00016873385424909302, 0.0002727008057159741, 0.0004948719241067732, 0.0009932669111481463,
        0.0018447476583195019, 0.0023181411138049925, 0.001808260941677265, 0.0011588441191995773, 0.0006295665857762873, 0.0002673675918108874,
        3.9811007530900256e-05, 4.21985638721677e-07, 4.98443369593083e-06, 4.4112686492917424e-05, 8.275340305853567e-05, 7.939933011064602e-05,
        5.434510733191397e-05, 4.1244828874587734e-05, 4.86518392479333e-05, 6.681203667241907e-05, 9.230860782794664e-05, 0.00014897956001500598,
        0.00024439293973494417, 0.0004327221442854445, 0.0008180539365098878, 0.001391906825501762, 0.0015471231516724286, 0.0009459587386766465,
        0.0004471929681118066, 0.00016897661876081482, 1.5442962277111285e-05, 8.547734665653016e-07, 1.6792760108705376e-05, 4.308466589904621e-05,
        3.901916811744249e-05, 2.0858331173180485e-05, 1.5273487684849824e-05, 1.7686113415584402e-05, 2.5990175312638637e-05, 3.637253470774397e-05,
        5.406059083723265e-05, 8.497642759443488e-05, 0.00012596453521472904, 0.00019437726636174703, 0.00035492255208756813, 0.0006600361285732164,
        0.00102443400944919, 0.0009732778823027861, 0.000420818025333731, 0.00010954374228635123, 5.627334167996607e-06, 5.844786157689087e-06,
        2.463684859028198e-05, 1.7884281746422013e-05, 7.638027841163088e-06, 5.275255520303315e-06, 5.8700419414317404e-06, 9.448479524389684e-06,
        1.4146463270657308e-05, 1.8974230721932213e-05, 2.833175497325692e-05, 4.1785819890252784e-05, 5.5343161169542923e-05, 7.328972934288015e-05,
        0.00011744556018502449, 0.0002175101606420203, 0.0003734597930480554, 0.0005232900038554059, 0.0003947430633796175, 9.109614813790043e-05,
        2.3617176617741875e-06, 7.436850267868821e-06, 9.79349898296524e-06, 3.560926462743726e-06, 2.2353162598556776e-06, 2.2270245580925396e-06,
        2.8910121388498914e-06, 5.4278258841174905e-06, 6.578665485504267e-06, 8.089477129422662e-06, 1.0783273102516525e-05, 1.4315098924312512e-05,
        1.617737072955451e-05, 1.7617238692504023e-05, 2.2242031768361204e-05, 3.430438213408536e-05, 5.673384993871574e-05, 9.241069023944079e-05,
        0.00013405434265967357, 5.918582958167397e-05, 3.787550660030711e-07, 4.5824286187103543e-07, 4.814657012936228e-07, 6.927228946408085e-07,
        1.0198476267987224e-06, 2.3483049607907144e-06, 2.925476796392643e-06, 2.8335363100211668e-06, 2.4896257134369032e-06, 2.081689303418293e-06,
        2.1228095537978733e-06, 1.664355963701127e-06, 1.289130950032941e-06, 1.1141332001824284e-06, 1.2292729125114294e-06, 1.5969706143326604e-06,
        2.4648962684954706e-06, 5.07245641923644e-06, 3.659735024894448e-06, 6.071444753897211e-09, 3.655595654195762e-08, 2.901575104914197e-07,
        1.4043046237453926e-06, 6.005115644322164e-07, 2.6490093993405136e-07, 1.209021548278971e-07, 4.785269100367103e-08, 2.88152931576716e-08,
        1.450395035269892e-08,
    ]
}

public extension NFKMLXColorizer {

    /// `colorization_train_val_v2.prototxt`'s batch, 40 images. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 40

    /// The prototxt's training crop, 176 pixels square. Introduced in InferKit 0.4.0.
    static let referenceImageSize = 176

    /// Builds the ECCV-16 colorizer for training or for reloading a trained checkpoint. With a `weightsURL`
    /// the released checkpoint or a file `NFKMLXWeights` saved loads; without one the network is randomly
    /// initialized, its annealed-mean readout set to the bin centers, which training holds. The network is in
    /// evaluation mode. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?) throws -> NFKMLXColorizerNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        } else {
            let readout = (NFKMLXColorizerObjective.binCenters / 110).transposed(1, 0).reshaped([2, 1, 1, 313])
            net.outAb.update(parameters: ModuleParameters.unflattened([("weight", readout)]))
        }
        return net
    }

    /// The network's input and target for RGB images `[N, H, W, 3]` in [0, 1], `H` and `W` multiples of 8:
    /// lightness normalized as `(L − 50) / 100`, `[N, H, W, 1]`, and the true ab at every fourth pixel,
    /// `[N, H/4, W/4, 2]`, as the prototxt's stride-4 `data_ab_ss` subsamples it. Introduced in InferKit 0.4.0.
    static func trainingExample(_ images: MLXArray) -> (lightness: MLXArray, ab: MLXArray) {
        let lab = NFKColorizerTrainingLab.reference(images)
        return ((lab[.ellipsis, 0 ..< 1] - 50) / 100, lab[0..., .stride(by: 4), .stride(by: 4), 1 ..< 3])
    }

    /// The optimizer `solver.prototxt` names: Caffe's Adam at 3.16e-5, momenta 0.9 and 0.99, delta 1e-8, its
    /// weight decay of 1e-3 joining every gradient. Caffe adds delta to the square root of the uncorrected
    /// second moment. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXKerasAdam(learningRate: 3.16e-5, beta1: 0.9, beta2: 0.99, epsilon: 1e-8, l2: 1e-3)
    }

    /// `solver.prototxt`'s step policy over a run of `steps`: the rate falls by 0.316 every 215,000 of 500,000
    /// iterations, placed at those fractions of the run and never at its first step. Introduced in
    /// InferKit 0.4.0.
    static func referenceSchedule(steps: Int) -> NFKMLXLearningRateSchedule {
        .multiStep(milestones: [max(steps * 215 / 500, 1), max(steps * 430 / 500, 1)], gamma: 0.316)
    }

    /// Fine-tunes the ECCV-16 colorizer on color images.
    ///
    /// - Parameters:
    ///   - net: the network, from network(weightsURL:).
    ///   - examples: supplies one batch per step: RGB images `[N, H, W, 3]` in [0, 1], `H` and `W` multiples
    ///     of 8. The reference crops 176 pixels square (referenceImageSize) with a random mirror; those are
    ///     the caller's.
    ///   - objective: the loss.
    ///   - optimizer: the update rule. Nil uses referenceOptimizer().
    ///   - steps: how many updates to train for.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference clips nothing.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is referenceSchedule(steps:) with the
    ///     reference optimizer and a constant rate with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every convolution trains; the normalizations' scales and offsets and the annealed-mean readout hold, as
    /// the prototxt's `lr_mult: 0` holds them, and the normalizations use each batch's statistics. Save with
    /// `NFKMLXWeights.save`; network(weightsURL:) and `backend(weightsURL:)` read the file back. A run is
    /// minutes; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXColorizerNet,
        examples: (Int) -> MLXArray,
        objective: NFKMLXColorizerObjective = NFKMLXColorizerObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let centers = NFKMLXColorizerObjective.binCenters
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                net.unfreeze()
                net.outAb.freeze()
                for case let norm as BatchNorm in net.children().flattened().map(\.1) {
                    norm.freeze()
                }
            },
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { referenceSchedule(steps: steps) },
            steps: steps,
            arrays: { step in
                let example = trainingExample(examples(step))
                return [example.lightness, example.ab]
            },
            loss: { net, arrays in objective.loss(logits: net.logits(arrays[0]), ab: arrays[1], binCenters: centers).rebalanced },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
