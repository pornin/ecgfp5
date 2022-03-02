# Curve ecGFp5

EcGFp5 is an elliptic curve defined over the field GF(p^5), for the
prime p = 2^64 - 2^32 + 1. This curve was designed to efficiently
support systems with a special compute model where operations in GF(p)
are especially efficient. One such system is the [Miden
VM](https://hackmd.io/YDbjUVHTRn64F4LPelC-NA), which is used to produce
and verify zero-knowledge proofs (of the
[STARK](https://eprint.iacr.org/2018/046) kind). EcGFp5 is not
inherently tied to Miden, but that compute model served as inspiration
for the curve choice.

  - The curve design and choice is explained in the
    [whitepaper](doc/ecgfp5.pdf). The paper is now also on
    [eprint](https://eprint.iacr.org/2022/274).

  - A test implementation, in Python, is provided in the [python](python/)
    directory. It emulates operations as they are meant to occur within
    the target compute model (the "VM") and counts costs, expressed in
    such operations.

  - A reference implementation in Rust (optimized and constant-time, but
    portable) is provided in the [rust](rust/) directory. It implements
    all operations on curve elements needed to implement, for instance,
    digital signatures.

# Curve Parameters

EcGFp5 is a [double-odd curve](https://doubleodd.group/): it is best
understood as a prime order group with efficient and constant-time group
operations, as well as canonical encoding and decoding procedures.

Base modulus is `p = 2^64 - 2^32 + 1 = 18446744069414584321`.

Field GF(p^5) is defined as polynomials in GF(p)[z], for a symbolic
variable z, of degree at most 4, and all computations performed
modulo the irreducible polynomial M = z^5 - 3.

The curve equation is: `y^2 = x*(x^2 + a*x + b)` for the two constants
`a = 2` and `b = 263*z`. The total curve order is `2*n` for a big
319-bit prime `n`:

    n = 1067993516717146951041484916571792702745057740581727230159139685185762082554198619328292418486241

EcGFp5 itself is defined as a group of order exactly `n`, consisting of
the points of the curve which are _not_ points of `n`-torsion. The
neutral element is the point `N = (0, 0)` (on the curve, this is the
unique point of order 2). The sum of two points `P` and `Q` in the
group is defined as `P + Q + N` on the curve. This yields a group with
all needed properties for defining cryptographic operations such as
key exchange or signatures:

  - The group has a prime order (no cofactor!).

  - Encoding and decoding of elements is canonical: each element can be
    encoded into a single sequence of bytes, and the decoding process
    succeeds only if provided that exact sequence of bytes.

  - The group law can be applied with complete formulas, with no
    special case; the formulas are furthermore efficient (generic
    addition in 10M, amortized per-doubling cost in 2M+5S).

The conventional generator `G` is the unique point of the group such
that `y/x = 4` (there are exactly two such points on the base curve,
but only one of them is in the prime-order group). Its coordinates
are:

    x = 12883135586176881569
      +  4356519642755055268*z
      +  5248930565894896907*z^2
      +  2165973894480315022*z^3
      +  2448410071095648785*z^4
    y = 4*x

As described in the [double-odd curves](https://doubleodd.group/),
there are several choices for internal coordinates. In general, the
_fractional (x,u)_ coordinates are recommended: an element is
represented as the quadruplet `(X:Z:U:T)` with `x = X/Z` and
`u = x/y = U/T` (with `u = 0` for the neutral `N`); elements `Z` and
`T` are never zero.

# Rust Implementation

The [rust](rust/) directory contains a reference implementation of
ecGFp5. It is portable: the code uses only `core` (not `std`) and has no
architecture-dependent features (no assembly, intrinsics or anything
tagged `unsafe`); it is nonetheless quite optimized. It does not require
the "nightly" compiler. The main types are `GFp` (integers modulo the
64-bit integer `p`), `GFp5` (elements of the GF(p^5) field), `Point`
(group elements) and `Scalar` (integers modulo the prime group order
`n`). The structures implement the usual traits (e.g. `Add` and
`AddAssign`) so that arithmetic operators can be used on such values.

The implementation is considered secure (notably, everything is
constant-time, except the functions meant to directly support signature
verification, which nominally uses only public data). It shall be noted
that there is no attempt whatsoever at "wiping memory", an action is
believed in some circles to be an important security feature. At least,
being a `core`-only implementation, there should be no dynamic memory
allocation except on the stack, so that a blanket stack-wiping strategy
after execution of curve-related code is feasible.

Compilation is straightforward (`cargo build` and so on). It can be
beneficial to performance to add explicit compilation flags in order to
leverage some architecture-specific opcodes, e.g. `mulx` on recent x86
CPUs (Intel Skylake and later); this is done by setting the `RUSTFLAGS`
environment variable with the value `-C target-cpu=native` (but this, of
course, prevents execution of the resulting binary on older CPUs that do
not feature such opcodes).

There are benchmarks (`cargo bench`). These benchmarks use the intrinsic
functions `_mm_lfence()` and `_rdtsc()` and will compile only on 64-bit
x86 architectures. Moreover, they use the CPU cycle counter, which is
known *not* to actually count cycles on CPUs that indulge in frequency
scaling (aka "TurboBoost"): if you run the benchmarks on a machine that
has TurboBoost enabled, the numbers you get are meaningless.

# Signatures

The provided implementations do _not_ include signature primitives. The
main reason is that defining a signature scheme entails choosing a hash
function, and it is expected that in-VM implementations will want to use
a specialized hash function such as [Poseidon](https://www.poseidon-hash.info/)
or [Rescue](https://www.esat.kuleuven.be/cosic/sites/rescue/).

Given the hash function `H` with an output size of at least 320 bits,
the _Schnorr signatures_ work as follows:

  - The private key is `d`, a non-zero scalar (integer modulo `n`).
    The public key is `Q = d*G`.

  - To sign a message `m` (which may itself be already a hash value):

     1. Generate a new random non-zero `k` modulo `n`. The choice must be
        uniform (or indinstinguishable from uniform) amoung all integers
        modulo `n`. If a secure random source is not available,
        derandomization techniques can be used, such as described in
        [RFC 6979](https://datatracker.ietf.org/doc/html/rfc6979).
        In this case, it may be convenient to hash together the private
        key, the public key and the message, and interpret the hash
        output as an integer, which is then reduced modulo `n` (assuming
        that the hash output is large enough that the reduction does not
        induce any notable bias; a 384-bit output is enough, since `n`
        has size 319 bits). In the Rust implementation,
        `Scalar::decode_reduce()` can perform the decoding and modular
        reduction.

     2. Compute point `R = k*G`. Since this uses the conventional generator,
        precomputed tables can be used to speed up the process; in the
        Rust code, use `Point::mulgen()`.

     3. Encode `R` into the 40-byte sequence `rbuf` (use `Point::encode()`
        to encode the point `R` into an element of GF(p^5), then
        `GFp5::encode() to obtain 40 bytes).

     4. Hash (with `H`) the concatenation of `rbuf`, the encoding of `Q`,
        and the message `m`. The hash output is interpreted as an integer
        (e.g. with little-endian convention) which is reduced modulo `n`,
        yielding the scalar `e`.

     5. Compute the scalar `s = k + d*e` (all computations modulo `n`;
        in Rust, use the arithmetic operations on `Scalar` instances).
        The scalar `s` is encoded into the 40-byte value `sbuf`
        (with `Scalar::encode()`).

     6. The signature is the concatenation of `rbuf` and `sbuf`.

  - To verify a signature over a message `m`, against a public key `Q`:

     1. Check that the signature has length exactly 80 bytes; split it
        into `rbuf` and `sbuf`.

     2. Decode `rbuf` into the point `R` (`GFp5::decode()`, then
        `Point::decode()`). Each decoding may fail, in case the value
        is invalid; a failed decoding implies that the signature is to
        be rejected.

     3. Decode `sbuf` into the scalar `s` (with `Scalar::decode()`, which
        may also fail if `sbuf` is not in the proper range).

     4. Hash the concatenation of `rbuf`, the encoding of `Q`, and the
        message `m`, and reduce the output modulo `n`. This recomputes
        the value `e` as in step 4 of the signature generation.

     5. Check that `s*G - e*Q = R`. In the Rust code, the function
        `Point::verify_muladd_vartime()` can be used (with scalars
        `s` and `-e`, and point `R`): this function applies an
        [optimized process](https://eprint.iacr.org/2020/454) to
        speed up this operation. As the name indicate, this function
        is _not_ constant-time, which is normally not a problem for
        signature verification, which uses only public data.
