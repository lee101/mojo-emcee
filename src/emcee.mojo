"""Numerical kernels for the affine-invariant ensemble sampler."""

from std.math import cos, log, sin
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FFT_PARALLEL_THRESHOLD = 131_072
comptime FFT_PARALLEL_TASKS = 16


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def stretch_proposal(
    sample: FPtr,
    complement: FPtr,
    scales: FPtr,
    partners: IPtr,
    proposal: FPtr,
    factors: FPtr,
    ns: Int,
    nc: Int,
    ndim: Int,
):
    for i in range(ns):
        var z = scales[i]
        var partner = Int(partners[i])
        if partner < 0:
            partner = 0
        if partner >= nc:
            partner = nc - 1
        var srow = i * ndim
        var crow = partner * ndim
        for d in range(ndim):
            var c = complement[crow + d]
            proposal[srow + d] = c - (c - sample[srow + d]) * z
        factors[i] = Float64(ndim - 1) * log(z)


def accept_proposals(
    coords: FPtr,
    log_prob: FPtr,
    proposal: FPtr,
    proposal_log_prob: FPtr,
    factors: FPtr,
    log_uniform: FPtr,
    indices: IPtr,
    accepted: IPtr,
    ns: Int,
    ndim: Int,
) -> Int:
    var count = 0
    for i in range(ns):
        var j = Int(indices[i])
        if factors[i] + proposal_log_prob[i] - log_prob[j] > log_uniform[i]:
            accepted[j] = 1
            log_prob[j] = proposal_log_prob[i]
            var src = i * ndim
            var dst = j * ndim
            for d in range(ndim):
                coords[dst + d] = proposal[src + d]
            count += 1
    return count


def bit_reverse(real: FPtr, imag: FPtr, n: Int):
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            var tr = real[i]
            real[i] = real[j]
            real[j] = tr
            var ti = imag[i]
            imag[i] = imag[j]
            imag[j] = ti


def fft_stage_parallel(
    real: FPtr,
    imag: FPtr,
    n: Int,
    width: Int,
    wr_step: Float64,
    wi_step: Float64,
):
    @parameter
    def process_blocks(task: Int):
        var first = task * (n // width) // FFT_PARALLEL_TASKS
        var last = (task + 1) * (n // width) // FFT_PARALLEL_TASKS
        for block in range(first, last):
            var base = block * width
            var wr = 1.0
            var wi = 0.0
            var half = width >> 1
            for k in range(half):
                var even = base + k
                var odd = even + half
                var tr = wr * real[odd] - wi * imag[odd]
                var ti = wr * imag[odd] + wi * real[odd]
                var er = real[even]
                var ei = imag[even]
                real[even] = er + tr
                imag[even] = ei + ti
                real[odd] = er - tr
                imag[odd] = ei - ti
                var next_wr = wr * wr_step - wi * wi_step
                wi = wr * wi_step + wi * wr_step
                wr = next_wr

    # `parallelize` moved out of the Mojo standard library in 1.1. Keep the
    # existing task partitioning while running the independent blocks here.
    for task in range(FFT_PARALLEL_TASKS):
        process_blocks(task)


def fft_in_place(real: FPtr, imag: FPtr, n: Int, inverse: Bool):
    bit_reverse(real, imag, n)
    var width = 2
    while width <= n:
        var angle = 6.283185307179586476925286766559 / Float64(width)
        if not inverse:
            angle = -angle
        var wr_step = cos(angle)
        var wi_step = sin(angle)
        if n >= FFT_PARALLEL_THRESHOLD:
            fft_stage_parallel(real, imag, n, width, wr_step, wi_step)
        else:
            var base = 0
            while base < n:
                var wr = 1.0
                var wi = 0.0
                var half = width >> 1
                for k in range(half):
                    var even = base + k
                    var odd = even + half
                    var tr = wr * real[odd] - wi * imag[odd]
                    var ti = wr * imag[odd] + wi * real[odd]
                    var er = real[even]
                    var ei = imag[even]
                    real[even] = er + tr
                    imag[even] = ei + ti
                    real[odd] = er - tr
                    imag[odd] = ei - ti
                    var next_wr = wr * wr_step - wi * wi_step
                    wi = wr * wi_step + wi * wr_step
                    wr = next_wr
                base += width
        width <<= 1


def autocorrelation_1d(
    values: FPtr,
    result: FPtr,
    work_real: FPtr,
    work_imag: FPtr,
    n: Int,
    fft_size: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var mean_vector = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        mean_vector += values.load[width=W](i)
        i += W
    var mean = mean_vector.reduce_add()
    while i < n:
        mean += values[i]
        i += 1
    mean /= Float64(n)

    i = 0
    var zeros = SIMD[DType.float64, W](0.0)
    while i + W <= n:
        work_real.store(i, values.load[width=W](i) - mean)
        work_imag.store(i, zeros)
        i += W
    while i < n:
        work_real[i] = values[i] - mean
        work_imag[i] = 0.0
        i += 1
    while i + W <= fft_size:
        work_real.store(i, zeros)
        work_imag.store(i, zeros)
        i += W
    while i < fft_size:
        work_real[i] = 0.0
        work_imag[i] = 0.0
        i += 1

    fft_in_place(work_real, work_imag, fft_size, False)
    i = 0
    while i + W <= fft_size:
        var re = work_real.load[width=W](i)
        var im = work_imag.load[width=W](i)
        work_real.store(i, re * re + im * im)
        work_imag.store(i, zeros)
        i += W
    while i < fft_size:
        work_real[i] = (
            work_real[i] * work_real[i] + work_imag[i] * work_imag[i]
        )
        work_imag[i] = 0.0
        i += 1
    fft_in_place(work_real, work_imag, fft_size, True)

    var norm = work_real[0]
    i = 0
    while i + W <= n:
        result.store(i, work_real.load[width=W](i) / norm)
        i += W
    while i < n:
        result[i] = work_real[i] / norm
        i += 1


@export("mem_stretch_proposal")
def mem_stretch_proposal(
    sample: Int,
    complement: Int,
    scales: Int,
    partners: Int,
    proposal: Int,
    factors: Int,
    ns: Int,
    nc: Int,
    ndim: Int,
) abi("C") -> Int:
    if (
        sample == 0
        or complement == 0
        or scales == 0
        or partners == 0
        or proposal == 0
        or factors == 0
        or ns <= 0
        or nc <= 0
        or ndim <= 0
    ):
        return -1
    var partner_ptr = ip(partners)
    for i in range(ns):
        var partner = Int(partner_ptr[i])
        if partner < 0 or partner >= nc:
            return -2
    stretch_proposal(
        fp(sample),
        fp(complement),
        fp(scales),
        ip(partners),
        fp(proposal),
        fp(factors),
        ns,
        nc,
        ndim,
    )
    return 0


@export("mem_accept_proposals")
def mem_accept_proposals(
    coords: Int,
    log_prob: Int,
    proposal: Int,
    proposal_log_prob: Int,
    factors: Int,
    log_uniform: Int,
    indices: Int,
    accepted: Int,
    ns: Int,
    ndim: Int,
) abi("C") -> Int:
    if (
        coords == 0
        or log_prob == 0
        or proposal == 0
        or proposal_log_prob == 0
        or factors == 0
        or log_uniform == 0
        or indices == 0
        or accepted == 0
        or ns < 0
        or ndim <= 0
    ):
        return -1
    return accept_proposals(
        fp(coords),
        fp(log_prob),
        fp(proposal),
        fp(proposal_log_prob),
        fp(factors),
        fp(log_uniform),
        ip(indices),
        ip(accepted),
        ns,
        ndim,
    )


@export("mem_autocorrelation_1d")
def mem_autocorrelation_1d(
    values: Int,
    result: Int,
    work_real: Int,
    work_imag: Int,
    n: Int,
    fft_size: Int,
) abi("C") -> Int:
    if (
        values == 0
        or result == 0
        or work_real == 0
        or work_imag == 0
        or n <= 0
        or fft_size < 2 * n
        or (fft_size & (fft_size - 1)) != 0
    ):
        return -1
    autocorrelation_1d(
        fp(values), fp(result), fp(work_real), fp(work_imag), n, fft_size
    )
    return 0
