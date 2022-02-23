#![allow(non_snake_case)]

use ecgfp5::scalar::Scalar;
use core::arch::x86_64::{_mm_lfence, _rdtsc};

fn core_cycles() -> u64 {
    unsafe {
        _mm_lfence();
        _rdtsc()
    }
}

// A custom PRNG; not cryptographically secure, but good enough
// for tests.
struct PRNG(u128);

impl PRNG {

    // A is a random prime. B is random.
    const A: u128 = 87981536952642681582438141175044346919;
    const B: u128 = 331203846847999889118488772711684568728;

    fn next_u64(&mut self) -> u64 {
        self.0 = PRNG::A.wrapping_mul(self.0).wrapping_add(PRNG::B);
        (self.0 >> 32) as u64
    }

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

fn bench_scalar_lagrange() {
    // Since Lagrange's algorithm execution time is variable, we need to
    // make some random inputs out-of-band.
    let mut prng = PRNG(core_cycles() as u128);
    let mut tt = [0; 10];
    for i in 0..10 {
        let mut kk = [Scalar::ZERO; 300];
        for j in 0..kk.len() {
            let mut kbuf = [0u8; 40];
            prng.next(&mut kbuf);
            kk[j] = Scalar::decode_reduce(&kbuf);
        }
        let begin = core_cycles();
        for j in 0..300 {
            let (c0, c1) = kk[j].lagrange();
            let tmp0 = c0.to_u192();
            let tmp1 = c1.to_u192();
            prng.0 = prng.0.wrapping_add(tmp0[0] as u128);
            prng.0 = prng.0.wrapping_add(tmp1[0] as u128);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let x = prng.next_u64();
    println!("Scalar lagrange: {:11.2}  ({})", (tt[4] as f64) / 300.0, x);
}

fn main() {
    bench_scalar_lagrange();
}
