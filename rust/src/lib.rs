#![no_std]

pub mod field;
pub mod curve;
pub mod scalar;
pub mod multab;

// A custom PRNG; not cryptographically secure, but good enough
// for tests.
#[cfg(test)]
struct PRNG(u128);

#[cfg(test)]
impl PRNG {
    // A: a randomly selected prime integer.
    // B: a randomly selected odd integer.
    const A: u128 = 87981536952642681582438141175044346919;
    const B: u128 = 331203846847999889118488772711684568729;

    // Get the next pseudo-random 64-bit integer.
    fn next_u64(&mut self) -> u64 {
        self.0 = PRNG::A.wrapping_mul(self.0).wrapping_add(PRNG::B);
        (self.0 >> 64) as u64
    }

    // Fill buf[] with pseudo-random bytes.
    fn next(&mut self, buf: &mut [u8]) {
        let mut acc: u64 = 0;
        for i in 0..buf.len() {
            if (i & 7) == 0 {
                acc = self.next_u64();
            }
            buf[i] = acc as u8;
            acc >>= 8;
        }
    }
}
