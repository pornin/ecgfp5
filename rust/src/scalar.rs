use core::ops::{Add, AddAssign, Neg, Sub, SubAssign, Mul, MulAssign};

/// A scalar (integer modulo the prime group order n).
#[derive(Clone, Copy, Debug)]
pub struct Scalar([u64; 5]);

impl Scalar {

    // IMPLEMENTATION NOTES:
    // ---------------------
    //
    // Group order n is slightly below 2^319. We store values over five
    // 64-bit limbs. We use Montgomery multiplication to perform
    // computations; however, we keep the limbs in normal
    // (non-Montgomery) representation, so that operations that do not
    // require any multiplication of scalars, just encoding and
    // decoding, are fastest.

    pub const ZERO: Self = Self([0, 0, 0, 0, 0]);

    pub const ONE: Self = Self([1, 0, 0, 0, 0]);

    // The modulus itself, stored in a Scalar structure (which
    // contravenes to the rules of a Scalar; this constant MUST NOT leak
    // outside the API).
    const N: Self = Self([
        0xE80FD996948BFFE1,
        0xE8885C39D724A09C,
        0x7FFFFFE6CFB80639,
        0x7FFFFFF100000016,
        0x7FFFFFFD80000007,
    ]);

    // -1/N[0] mod 2^64
    const N0I: u64 = 0xD78BEF72057B7BDF;

    /* not used
    // 2^320 mod n.
    const R: Self = Self([
        0x2FE04CD2D6E8003E,
        0x2EEF478C51B6BEC6,
        0x00000032608FF38C,
        0x0000001DFFFFFFD3,
        0x00000004FFFFFFF1,
    ]);
    */

    // 2^640 mod n.
    const R2: Self = Self([
        0xA01001DCE33DC739,
        0x6C3228D33F62ACCF,
        0xD1D796CC91CF8525,
        0xAADFFF5D1574C1D8,
        0x4ACA13B28CA251F5,
    ]);

    // 2^632 mod n.
    const T632: Self = Self([
        0x2B0266F317CA91B3,
        0xEC1D26528E984773,
        0x8651D7865E12DB94,
        0xDA2ADFF5941574D0,
        0x53CACA12110CA256,
    ]);

    // raw addition (no reduction)
    fn add_inner(self, a: Self) -> Self {
        let mut r = Self::ZERO;
        let mut c: u64 = 0;
        for i in 0..5 {
            let z = (self.0[i] as u128).wrapping_add(a.0[i] as u128)
                .wrapping_add(c as u128);
            r.0[i] = z as u64;
            c = (z >> 64) as u64;
        }
        // no extra carry, since inputs are supposed to fit on 319 bits.
        r
    }

    // raw subtraction (no reduction)
    // Final borrow is returned (0xFFFFFFFFFFFFFFFF if borrow, 0 otherwise).
    fn sub_inner(self, a: Self) -> (Self, u64) {
        let mut r = Self::ZERO;
        let mut c: u64 = 0;
        for i in 0..5 {
            let z = (self.0[i] as u128).wrapping_sub(a.0[i] as u128)
                .wrapping_sub(c as u128);
            r.0[i] = z as u64;
            c = ((z >> 64) as u64) & 1;
        }
        (r, c.wrapping_neg())
    }

    /// If c == 0, return a0.
    /// If c == 0xFFFFFFFFFFFFFFFF, return a1.
    /// c MUST be equal to 0 or 0xFFFFFFFFFFFFFFFF.
    pub fn select(c: u64, a0: Self, a1: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..5 {
            r.0[i] = a0.0[i] ^ (c & (a0.0[i] ^ a1.0[i]));
        }
        r
    }

    // Scalar addition.
    fn add(self, rhs: Self) -> Self {
        let r0 = self.add_inner(rhs);
        let (r1, c) = r0.sub_inner(Self::N);
        Self::select(c, r1, r0)
    }

    // Scalar subtraction.
    fn sub(self, rhs: Self) -> Self {
        let (r0, c) = self.sub_inner(rhs);
        let r1 = r0.add_inner(Self::N);
        Self::select(c, r0, r1)
    }

    // Scalar negation.
    fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    // Montgomery multiplication.
    // Returns (self*rhs)/2^320 mod n.
    // 'self' MUST be less than n (the other operand can be up to 2^320-1).
    fn montymul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..5 {
            // Iteration i computes r <- (r + self*rhs_i + f*n)/2^64.
            // Factor f is at most 2^64-1 and set so that the division
            // is exact.
            // On input:
            //    r <= 2^320 - 1
            //    self <= n - 1
            //    rhs_i <= 2^64 - 1
            //    f <= 2^64 - 1
            // Therefore:
            //    r + self*rhs_i + f*n <= 2^320-1 + (2^64 - 1) * (n - 1)
            //                            + (2^64 - 1) * n
            //                         < 2^384
            // Thus, the new r fits on 320 bits.
            let m = rhs.0[i];
            let f = self.0[0].wrapping_mul(m).wrapping_add(r.0[0])
                .wrapping_mul(Self::N0I);
            let mut cc1: u64 = 0;
            let mut cc2: u64 = 0;
            for j in 0..5 {
                let mut z = (self.0[j] as u128).wrapping_mul(m as u128)
                    .wrapping_add(r.0[j] as u128).wrapping_add(cc1 as u128);
                cc1 = (z >> 64) as u64;
                z = (f as u128).wrapping_mul(Self::N.0[j] as u128)
                    .wrapping_add((z as u64) as u128).wrapping_add(cc2 as u128);
                cc2 = (z >> 64) as u64;
                if j > 0 {
                    r.0[j - 1] = z as u64;
                }
            }
            // No overflow here since the new r fits on 320 bits.
            r.0[4] = cc1.wrapping_add(cc2);
        }

        // We computed (self*rhs + ff*n) / 2^320, with:
        //    self < n
        //    rhs < 2^320
        //    ff < 2^320
        // Thus, the value we obtained is lower than 2*n. Subtracting n
        // once (conditionally) is sufficient to achieve full reduction.
        let (r2, c) = r.sub_inner(Self::N);
        Self::select(c, r2, r)
    }

    fn mul(self, rhs: Self) -> Self {
        self.montymul(Self::R2).montymul(rhs)
    }

    /// Decode the provided byte slice into a scalar. The bytes are
    /// interpreted into an integer in little-endian unsigned convention.
    /// All slice bytes are read. Return value is (s, c):
    ///  - If the decoded integer is lower than the group order, then that
    ///    value is returned as s, and c == 0xFFFFFFFFFFFFFFFF.
    ///  - Otherwise, s is set to Scalar::ZERO, and c == 0.
    pub fn decode(buf: &[u8]) -> (Self, u64) {
        let n = buf.len();
        let mut r = Self::ZERO;
        let mut extra: u8 = 0;
        for i in 0..n {
            if i < 40 {
                r.0[i >> 3] |= (buf[i] as u64)
                    .wrapping_shl(((i as u32) & 7) << 3);
            } else {
                extra |= buf[i];
            }
        }

        // If input buffer is at most 39 bytes then the result is
        // necessarily in range; we can skip the reduction tests.
        if n <= 39 {
            return (r, 0xFFFFFFFFFFFFFFFF);
        }

        // Output is in the correct range if and only if extra == 0 and
        // the value is lower than n.
        let (_, mut c) = r.sub_inner(Self::N);
        c &= ((extra as u64).wrapping_add(0xFF) >> 8).wrapping_sub(1);
        for i in 0..5 {
            r.0[i] &= c;
        }
        (r, c)
    }

    /// Decode the provided byte slice into a scalar. The bytes are
    /// interpreted into an integer in little-endian unsigned convention.
    /// All slice bytes are read, and the value is REDUCED modulo n. This
    /// function never fails; it accepts arbitrary input values.
    pub fn decode_reduce(buf: &[u8]) -> Self {
        // We inject the value by chunks of 312 bits, in high-to-low
        // order. We multiply by 2^312 the intermediate result, which
        // is equivalent to performing a Montgomery multiplication
        // by 2^632 mod n.

        // If buffer length is at most 39 bytes, then the plain decode()
        // function works.
        let n = buf.len();
        if n <= 39 {
            let (r, _) = Self::decode(buf);
            return r;
        }

        // We can now assume that we have at least 40 bytes of input.

        // Compute k as a multiple of 39 such that n-39 <= k < n. Since
        // n >= 40, this implies that k >= 1. We decode the top chunk
        // (which has length _at most_ 39 bytes) into acc.
        let mut k = ((n - 1) / 39) * 39;
        let (mut acc, _) = Self::decode(&buf[k..n]);
        while k > 0 {
            k -= 39;
            let (b, _) = Self::decode(&buf[k..k+39]);
            acc = acc.montymul(Self::T632).add(b);
        }
        acc
    }

    /// Encode this scalar over exactly 40 bytes.
    pub fn encode(self) -> [u8; 40] {
        let mut r = [0u8; 40];
        for i in 0..5 {
            r[8*i..8*i+8].copy_from_slice(&self.0[i].to_le_bytes());
        }
        r
    }

    // Recode a scalar into signed integers. For a window width of w
    // bits, returned integers are in the -(2^w-1) to +2^w range. The
    // provided slice is filled; if w*len(ss) >= 320, then the output
    // encodes the complete scalar value, and the top (last) signed
    // integer is nonnegative.
    // Window width MUST be between 2 and 10.
    pub(crate) fn recode_signed(self, ss: &mut [i32], w: i32) {
        Self::recode_signed_from_limbs(&self.0, ss, w);
    }

    fn recode_signed_from_limbs(limbs: &[u64], ss: &mut [i32], w: i32) {
        let mut acc: u64 = 0;
        let mut acc_len: i32 = 0;
        let mut j = 0;
        let mw = (1u32 << w) - 1;
        let hw = 1u32 << (w - 1);
        let mut cc: u32 = 0;
        for i in 0..ss.len() {
            // Get next w-bit chunk in bb.
            let mut bb: u32;
            if acc_len < w {
                if j < limbs.len() {
                    let nl = limbs[j];
                    j += 1;
                    bb = ((acc | (nl << acc_len)) as u32) & mw;
                    acc = nl >> (w - acc_len);
                } else {
                    bb = (acc as u32) & mw;
                    acc = 0;
                }
                acc_len += 64 - w;
            } else {
                bb = (acc as u32) & mw;
                acc_len -= w;
                acc >>= w;
            }

            // If bb is greater than 2^(w-1), subtract 2^w and
            // propagate a carry.
            bb += cc;
            cc = hw.wrapping_sub(bb) >> 31;
            ss[i] = (bb as i32).wrapping_sub((cc << w) as i32);
        }
    }

    // Use Lagrange's algorithm to represent this scalar k as a
    // pair (v0, v1) such that k = v0/v1 mod n.
    // This function is NOT constant-time and should be used only on
    // a non-secret scalar (e.g. as part of signature verification).
    pub fn lagrange(self) -> (Signed161, Signed161) {
        // We use algorithm 4 from: https://eprint.iacr.org/2020/454

        // Nu <- n^2
        // Nv <- k^2 + 1
        // sp <- n*k
        let mut nu_buf = Signed640::from_nsquared();
        let mut nv_buf = Signed640::from_mul_scalars(self, self);
        nv_buf.add1();
        let (mut nu, mut nv) = (&mut nu_buf, &mut nv_buf);
        let mut sp = Signed640::from_mul_scalars(self, Self::N);

        // (u0, u1) <- (n, 0)
        // (v0, v1) <- (k, 1)
        let mut u0_buf = Signed161::from_scalar(Self::N);
        let mut u1_buf = Signed161::from_scalar(Self::ZERO);
        let mut v0_buf = Signed161::from_scalar(self);
        let mut v1_buf = Signed161::from_scalar(Self::ONE);
        let (mut u0, mut u1) = (&mut u0_buf, &mut u1_buf);
        let (mut v0, mut v1) = (&mut v0_buf, &mut v1_buf);

        // Main loop.
        loop {
            // if u is smaller than v, then swap them.
            if nu.lt_unsigned(nv) {
                let tn = nu;
                nu = nv;
                nv = tn;
                let (t0, t1) = (u0, u1);
                u0 = v0;
                u1 = v1;
                v0 = t0;
                v1 = t1;
            }

            // if len(Nv) <= 320, then we are finished.
            let vlen = nv.bitlength();
            if vlen <= 320 {
                return (*v0, *v1);
            }

            // shift count s = max(0, len(p) - len(Nv))
            let mut s = sp.bitlength() - vlen;
            if s < 0 {
                s = 0;
            }

            if sp.is_nonnegative() {
                u0.sub_shifted(v0, s);
                u1.sub_shifted(v1, s);
                nu.add_shifted(nv, s << 1);
                nu.sub_shifted(&sp, s + 1);
                sp.sub_shifted(nv, s);
            } else {
                u0.add_shifted(v0, s);
                u1.add_shifted(v1, s);
                nu.add_shifted(nv, s << 1);
                nu.add_shifted(&sp, s + 1);
                sp.add_shifted(nv, s);
            }
        }
    }

    /// Compare this scalar with zero. Returned value is 0xFFFFFFFFFFFFFFFF
    /// if this scalar is zero, or 0 otherwise.
    pub fn iszero(self) -> u64 {
        let x = self.0[0] | self.0[1] | self.0[2] | self.0[3] | self.0[4];
        ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
    }

    /// Compare this scalar with another one. Returned value is
    /// 0xFFFFFFFFFFFFFFFF if they are equal, or 0 otherwise.
    /// Equality is defined modulo n.
    pub fn equals(self, rhs: Self) -> u64 {
        let x = (self.0[0] ^ rhs.0[0])
            | (self.0[1] ^ rhs.0[1])
            | (self.0[2] ^ rhs.0[2])
            | (self.0[3] ^ rhs.0[3])
            | (self.0[4] ^ rhs.0[4]);
        ((x | x.wrapping_neg()) >> 63).wrapping_sub(1)
    }
}

impl Add<Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, other: Scalar) -> Scalar {
        Scalar::add(self, other)
    }
}

impl AddAssign<Scalar> for Scalar {
    fn add_assign(&mut self, other: Scalar) {
        *self = Scalar::add(*self, other)
    }
}

impl Sub<Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, other: Scalar) -> Scalar {
        Scalar::sub(self, other)
    }
}

impl SubAssign<Scalar> for Scalar {
    fn sub_assign(&mut self, other: Scalar) {
        *self = Scalar::sub(*self, other)
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Scalar {
        Scalar::neg(self)
    }
}

impl Mul<Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, other: Scalar) -> Scalar {
        Scalar::mul(self, other)
    }
}

impl MulAssign<Scalar> for Scalar {
    fn mul_assign(&mut self, other: Scalar) {
        *self = Scalar::mul(*self, other)
    }
}

/// A custom 161-bit integer type; used for splitting a scalar into a
/// fraction. Negative values use two's complement notation; the value
/// is truncated to 161 bits (upper bits in the top limb are ignored).
/// Elements are mutable containers.
/// WARNING: everything in here is vartime; do not use on secret values.
#[derive(Clone, Copy, Debug)]
pub struct Signed161([u64; 3]);

impl Signed161 {

    fn from_scalar(s: Scalar) -> Self {
        Self([s.0[0], s.0[1], s.0[2]])
    }

    /// Convert that value into a scalar (integer modulo n).
    pub fn to_scalar_vartime(self) -> Scalar {
        let mut tmp = self.to_u192();
        let neg = (tmp[2] >> 63) != 0;
        if neg {
            tmp[0] = (!tmp[0]).wrapping_add(1);
            let mut cc = tmp[0] == 0;
            tmp[1] = !tmp[1];
            if cc {
                tmp[1] = tmp[1].wrapping_add(1);
                cc = tmp[1] == 0;
            }
            tmp[2] = !tmp[2];
            if cc {
                tmp[2] = tmp[2].wrapping_add(1);
            }
            return -Scalar([tmp[0], tmp[1], tmp[2], 0, 0]);
        } else {
            return Scalar([tmp[0], tmp[1], tmp[2], 0, 0]);
        }
    }

    /// Export this value as a 192-bit integer (three 64-bit limbs,
    /// in little-endian order).
    pub fn to_u192(self) -> [u64; 3] {
        let mut x = self.0[2];
        x &= 0x00000001FFFFFFFF;
        x |= (x >> 32).wrapping_neg() << 33;
        [self.0[0], self.0[1], x]
    }

    // Recode this integer into 33 signed digits for a 5-bit window.
    pub(crate) fn recode_signed_5(self) -> [i32; 33] {
        // We first sign-extend the value to 192 bits, then add
        // 2^160 to get a nonnegative value in the 0 to 2^161-1
        // range. We then recode that value; and finally we fix
        // the result by subtracting 1 from the top digit.
        let mut tmp = self.to_u192();
        tmp[2] = tmp[2].wrapping_add(0x0000000100000000);
        let mut ss = [0i32; 33];
        Scalar::recode_signed_from_limbs(&tmp, &mut ss, 5);
        ss[32] -= 1;
        ss
    }

    // Add v*2^s to this value.
    fn add_shifted(&mut self, v: &Signed161, s: i32) {
        if s == 0 {
            Self::add(self, &v.0[..]);
        } else if s < 64 {
            Self::add_shifted_small(self, &v.0[..], s);
        } else if s < 161 {
            Self::add_shifted_small(self, &v.0[((s >> 6) as usize)..], s & 63);
        }
    }

    fn add_shifted_small(&mut self, v: &[u64], s: i32) {
        let mut cc = 0u64;
        let j = 3 - v.len();
        let mut vbits = 0u64;
        for i in j..3 {
            let vw = v[i - j];
            let vws = vw.wrapping_shl(s as u32) | vbits;
            vbits = vw.wrapping_shr((64 - s) as u32);
            let z = (self.0[i] as u128) + (vws as u128) + (cc as u128);
            self.0[i] = z as u64;
            cc = (z >> 64) as u64;
        }
    }

    fn add(&mut self, v: &[u64]) {
        let mut cc = 0;
        let j = 3 - v.len();
        for i in j..3 {
            let z = (self.0[i] as u128) + (v[i - j] as u128) + (cc as u128);
            self.0[i] = z as u64;
            cc = (z >> 64) as u64;
        }
    }

    // Subtract v*2^s from this value.
    fn sub_shifted(&mut self, v: &Signed161, s: i32) {
        if s == 0 {
            Self::sub(self, &v.0[..]);
        } else if s < 64 {
            Self::sub_shifted_small(self, &v.0[..], s);
        } else if s < 161 {
            Self::sub_shifted_small(self, &v.0[((s >> 6) as usize)..], s & 63);
        }
    }

    fn sub_shifted_small(&mut self, v: &[u64], s: i32) {
        let mut cc = 0u64;
        let j = 3 - v.len();
        let mut vbits = 0u64;
        for i in j..3 {
            let vw = v[i - j];
            let vws = vw.wrapping_shl(s as u32) | vbits;
            vbits = vw.wrapping_shr((64 - s) as u32);
            let z = (self.0[i] as u128).wrapping_sub(vws as u128)
                .wrapping_sub(cc as u128);
            self.0[i] = z as u64;
            cc = ((z >> 64) as u64) & 1;
        }
    }

    fn sub(&mut self, v: &[u64]) {
        let mut cc = 0;
        let j = 3 - v.len();
        for i in j..3 {
            let z = (self.0[i] as u128).wrapping_sub(v[i - j] as u128)
                .wrapping_sub(cc as u128);
            self.0[i] = z as u64;
            cc = ((z >> 64) as u64) & 1;
        }
    }
}

// A custom 640-bit integer type (signed).
// Elements are mutable containers.
// WARNING: everything in here is vartime; do not use on secret values.
#[derive(Clone, Copy, Debug)]
struct Signed640([u64; 10]);

impl Signed640 {

    // Obtain an instance containing n^2.
    fn from_nsquared() -> Self {
        Signed640([
            0x8E6B7A18061803C1,
            0x0AD8BDEE1594E2CF,
            0x17640E465F2598BC,
            0x90465B4214B27B1C,
            0xD308FECCB1878B88,
            0x3CC55EB2EAC07502,
            0x59F038FB784335CE,
            0xBFFFFE954FB808EA,
            0xBFFFFFCB80000099,
            0x3FFFFFFD8000000D,
        ])
    }

    // Obtain an instance containing a*b (both a and b are interpreted
    // as integers in the 0..n-1 range).
    fn from_mul_scalars(a: Scalar, b: Scalar) -> Self {
        let mut r = Signed640([0u64; 10]);
        for i in 0..5 {
            let aw = a.0[i];
            let mut cc = 0u64;
            for j in 0..5 {
                let bw = b.0[j];
                let z = ((aw as u128) * (bw as u128))
                    .wrapping_add(r.0[i + j] as u128)
                    .wrapping_add(cc as u128);
                r.0[i + j] = z as u64;
                cc = (z >> 64) as u64;
            }
            r.0[i + 5] = cc;
        }
        r
    }

    // Add 1 to this instance.
    fn add1(&mut self) {
        for i in 0..10 {
            self.0[i] = self.0[i].wrapping_add(1);
            if self.0[i] != 0 {
                return;
            }
        }
    }

    fn is_nonnegative(&self) -> bool {
        (self.0[9] >> 63) == 0
    }

    fn lt_unsigned(&self, rhs: &Self) -> bool {
        for i in (0..10).rev() {
            let aw = self.0[i];
            let bw = rhs.0[i];
            if aw < bw {
                return true;
            }
            if aw > bw {
                return false;
            }
        }
        false
    }

    // Get the bit length of this value. The bit length is defined as the
    // minimal size of the binary representation in two's complement,
    // _excluding_ the sign bit (thus, -2^k has bit length k, whereas +2^k
    // has bit length k+1).
    fn bitlength(&self) -> i32 {
        let sm = (self.0[9] >> 63).wrapping_neg();
        for i in (0..10).rev() {
            let w = self.0[i] ^ sm;
            if w != 0 {
                return ((i as i32) << 6) + Self::u64_bitlength(w);
            }
        }
        0
    }

    fn u64_bitlength(w: u64) -> i32 {
        // We use here a portable algorithm; some architectures have
        // dedicated opcodes that could speed up this operation
        // greatly (e.g. lzcnt on recent x86).
        let mut x = w;
        let mut r = 0;
        if x > 0xFFFFFFFF { x >>= 32; r += 32; }
        if x > 0x0000FFFF { x >>= 16; r += 16; }
        if x > 0x000000FF { x >>=  8; r +=  8; }
        if x > 0x0000000F { x >>=  4; r +=  4; }
        if x > 0x00000003 { x >>=  2; r +=  2; }
        r + (x as i32) - (((x + 1) >> 2) as i32)
    }

    // Add v*2^s to this instance.
    fn add_shifted(&mut self, v: &Signed640, s: i32) {
        if s == 0 {
            Self::add(self, &v.0[..]);
        } else if s < 64 {
            Self::add_shifted_small(self, &v.0[..], s);
        } else if s < 640 {
            Self::add_shifted_small(self, &v.0[((s >> 6) as usize)..], s & 63);
        }
    }

    fn add_shifted_small(&mut self, v: &[u64], s: i32) {
        let mut cc = 0u64;
        let j = 10 - v.len();
        let mut vbits = 0u64;
        for i in j..10 {
            let vw = v[i - j];
            let vws = vw.wrapping_shl(s as u32) | vbits;
            vbits = vw.wrapping_shr((64 - s) as u32);
            let z = (self.0[i] as u128) + (vws as u128) + (cc as u128);
            self.0[i] = z as u64;
            cc = (z >> 64) as u64;
        }
    }

    fn add(&mut self, v: &[u64]) {
        let mut cc = 0;
        let j = 10 - v.len();
        for i in j..10 {
            let z = (self.0[i] as u128) + (v[i - j] as u128) + (cc as u128);
            self.0[i] = z as u64;
            cc = (z >> 64) as u64;
        }
    }

    // Subtract v*2^s from this instance.
    fn sub_shifted(&mut self, v: &Signed640, s: i32) {
        if s == 0 {
            Self::sub(self, &v.0[..]);
        } else if s < 64 {
            Self::sub_shifted_small(self, &v.0[..], s);
        } else {
            Self::sub_shifted_small(self, &v.0[((s >> 6) as usize)..], s & 63);
        }
    }

    fn sub_shifted_small(&mut self, v: &[u64], s: i32) {
        let mut cc = 0u64;
        let j = 10 - v.len();
        let mut vbits = 0u64;
        for i in j..10 {
            let vw = v[i - j];
            let vws = vw.wrapping_shl(s as u32) | vbits;
            vbits = vw.wrapping_shr((64 - s) as u32);
            let z = (self.0[i] as u128).wrapping_sub(vws as u128)
                .wrapping_sub(cc as u128);
            self.0[i] = z as u64;
            cc = ((z >> 64) as u64) & 1;
        }
    }

    fn sub(&mut self, v: &[u64]) {
        let mut cc = 0;
        let j = 10 - v.len();
        for i in j..10 {
            let z = (self.0[i] as u128).wrapping_sub(v[i - j] as u128)
                .wrapping_sub(cc as u128);
            self.0[i] = z as u64;
            cc = ((z >> 64) as u64) & 1;
        }
    }
}

// ========================================================================
// Unit tests.

#[cfg(test)]
mod tests {
    use super::Scalar;
    use super::super::PRNG;

    #[test]
    fn scalar_ops() {
        let buf1: [u8; 50] = [
            0xE0, 0xFF, 0x8B, 0x94, 0x96, 0xD9, 0x0F, 0xE8,
            0x9C, 0xA0, 0x24, 0xD7, 0x39, 0x5C, 0x88, 0xE8,
            0x39, 0x06, 0xB8, 0xCF, 0xE6, 0xFF, 0xFF, 0x7F,
            0x16, 0x00, 0x00, 0x00, 0xF1, 0xFF, 0xFF, 0x7F,
            0x07, 0x00, 0x00, 0x80, 0xFD, 0xFF, 0xFF, 0x7F,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let buf2: [u8; 50] = [
            0xE1, 0xFF, 0x8B, 0x94, 0x96, 0xD9, 0x0F, 0xE8,
            0x9C, 0xA0, 0x24, 0xD7, 0x39, 0x5C, 0x88, 0xE8,
            0x39, 0x06, 0xB8, 0xCF, 0xE6, 0xFF, 0xFF, 0x7F,
            0x16, 0x00, 0x00, 0x00, 0xF1, 0xFF, 0xFF, 0x7F,
            0x07, 0x00, 0x00, 0x80, 0xFD, 0xFF, 0xFF, 0x7F,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let buf3: [u8; 50] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00,
        ];
        for i in 0..51 {
            let (s1, c1) = Scalar::decode(&buf1[..i]);
            let (s2, c2) = Scalar::decode(&buf2[..i]);
            let (s3, c3) = Scalar::decode(&buf3[..i]);
            assert!(c1 == 0xFFFFFFFFFFFFFFFF);
            if i <= 40 {
                assert!(s1.encode()[..i] == buf1[..i]);
            } else {
                assert!(s1.encode()[..] == buf1[..40]);
            }
            if i <= 39 {
                assert!(c2 == 0xFFFFFFFFFFFFFFFF);
                assert!(s2.encode()[..i] == buf2[..i]);
            } else {
                assert!(c2 == 0);
            }
            if i <= 47 {
                assert!(c3 == 0xFFFFFFFFFFFFFFFF);
                if i <= 40 {
                    assert!(s3.encode()[..i] == buf3[..i]);
                } else {
                    assert!(s3.encode()[..] == buf3[..40]);
                }
            } else {
                assert!(c3 == 0);
            }
        }

        // buf4 = a randomly chosen 512-bit integer
        let buf4: [u8; 64] = [
            0xB5, 0xDD, 0x28, 0xB8, 0xD2, 0x9B, 0x6F, 0xF8,
            0x15, 0x65, 0x3F, 0x89, 0xDB, 0x7B, 0xA9, 0xDE,
            0x33, 0x7D, 0xA8, 0x27, 0x82, 0x26, 0xB4, 0xD6,
            0x9E, 0x1F, 0xFA, 0x97, 0x3D, 0x9E, 0x01, 0x9C,
            0x77, 0xC9, 0x63, 0x5C, 0xB8, 0x34, 0xD8, 0x1A,
            0x4D, 0xCB, 0x03, 0x48, 0x62, 0xCD, 0xEE, 0xC9,
            0x8E, 0xC8, 0xC9, 0xA7, 0xB3, 0x6E, 0xDA, 0xCE,
            0x18, 0x75, 0x1B, 0xDD, 0x4F, 0x94, 0x67, 0xB5,
        ];
        // buf5 = buf4 mod n
        let buf5: [u8; 40] = [
            0x89, 0x01, 0x7A, 0x52, 0xBD, 0xDF, 0x45, 0x60,
            0xCE, 0x5B, 0xBA, 0xE5, 0x5D, 0x25, 0x96, 0x5A,
            0x0A, 0x4F, 0x0A, 0x27, 0x1A, 0x7A, 0xE8, 0x1D,
            0x7D, 0xBF, 0xE3, 0xE3, 0xFA, 0x5E, 0x17, 0xE0,
            0x44, 0xD9, 0xA5, 0x37, 0x9B, 0xF8, 0x38, 0x74,
        ];
        let s4 = Scalar::decode_reduce(&buf4[..]);
        assert!(s4.encode() == buf5);
        let (s5, c5) = Scalar::decode(&buf5[..]);
        assert!(c5 == 0xFFFFFFFFFFFFFFFF);
        assert!(s5.encode() == buf5);

        // buf6 = (buf4^256) mod n
        let buf6: [u8; 40] = [
            0x27, 0x7E, 0x2C, 0xAB, 0x6D, 0xAD, 0x8D, 0xA0,
            0x15, 0x44, 0x02, 0x0F, 0xFA, 0xD5, 0x4F, 0x15,
            0xBF, 0x6D, 0x1D, 0x76, 0x22, 0x73, 0xCD, 0xDA,
            0x23, 0xFE, 0x5A, 0xED, 0xCA, 0x75, 0xD7, 0x04,
            0x05, 0x66, 0x87, 0x3D, 0x37, 0x5B, 0x24, 0x13,
        ];
        let mut s6 = s4;
        for _ in 0..8 {
            s6 *= s6;
        }
        assert!(s6.encode() == buf6);

        // buf6 recoded in signed integers, w = 4
        let ref4: [i32; 80] = [
             7,  2, -2,  8, -4,  3, -5, -5, -2,  7, -3, -5, -2, -7,  1, -6,
             6,  1,  4,  4,  2,  0, -1,  1, -6,  0,  6, -3,  0,  5,  5,  1,
            -1, -4, -2,  7, -3,  2,  6,  7,  2,  2,  3,  7, -3, -3, -5, -2,
             4,  2, -2,  0, -5,  6, -3, -1, -5, -3,  6,  7,  7, -3,  5,  0,
             5,  0,  6,  6,  7,  8, -3,  4,  7,  3, -5,  6,  4,  2,  3,  1,
        ];
        // buf6 recoded in signed integers, w = 5
        let ref5: [i32; 64] = [
              7, -15,   0,  -7, -13, -10,  -9,  14,  13,  13,   3,   1,
             -6,  11,  16,   8,   2,  -8,   4, -12,   0,  11,  -1,  10,
            -11,  -7,  16,  -5,  -9,  15,  -8,  15,   2,  -7,  -3,  -5,
             13,  13,  15,   4,  -2,  -8,  -9,  -5,  15,   5,  -9,  15,
             -9,   7,   1,  10,   0, -13,  -2, -15,  -2,  -6,  14, -10,
              6, -14,  13,   2,
        ];

        let mut ss4 = [0i32; 80];
        s6.recode_signed(&mut ss4[..], 4);
        assert!(ss4 == ref4);
        let mut ss5 = [0i32; 64];
        s6.recode_signed(&mut ss5[..], 5);
        assert!(ss5 == ref5);
    }

    #[test]
    fn lagrange() {
        let mut prng = PRNG(0);
        for _ in 0..100 {
            let mut sbuf = [0u8; 48];
            prng.next(&mut sbuf);
            let s = Scalar::decode_reduce(&mut sbuf);
            let (v0, v1) = s.lagrange();
            let c0 = v0.to_scalar_vartime();
            let c1 = v1.to_scalar_vartime();
            assert!((c1 * s - c0).iszero() == 0xFFFFFFFFFFFFFFFF);
        }
    }
}
