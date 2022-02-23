#![allow(non_snake_case)]

use ecgfp5::curve::Point;
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

fn bench_curve_add() {
    let mut P = Point::GENERATOR;
    let mut ebuf = [0u8; 40];
    for i in 0..ebuf.len() {
        ebuf[i] = core_cycles() as u8;
    }
    let mut Q = P * Scalar::decode_reduce(&ebuf);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            Q += P;
            P += Q;
            Q += P;
            P += Q;
            Q += P;
            P += Q;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let x = P.encode().0[0];
    println!("Curve add:       {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_curve_double() {
    let mut P = Point::GENERATOR;
    let mut ebuf = [0u8; 40];
    for i in 0..ebuf.len() {
        ebuf[i] = core_cycles() as u8;
    }
    P *= Scalar::decode_reduce(&ebuf);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            P = P.double();
            P = P.double();
            P = P.double();
            P = P.double();
            P = P.double();
            P = P.double();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let x = P.encode().0[0];
    println!("Curve double:    {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_curve_double_x5() {
    let mut P = Point::GENERATOR;
    let mut ebuf = [0u8; 40];
    for i in 0..ebuf.len() {
        ebuf[i] = core_cycles() as u8;
    }
    P *= Scalar::decode_reduce(&ebuf);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            P = P.mdouble(5);
            P = P.mdouble(5);
            P = P.mdouble(5);
            P = P.mdouble(5);
            P = P.mdouble(5);
            P = P.mdouble(5);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let x = P.encode().0[0];
    println!("Curve double x5: {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_curve_mul() {
    let mut P = Point::GENERATOR;
    let mut ebuf = [0u8; 40];
    for i in 0..ebuf.len() {
        ebuf[i] = core_cycles() as u8;
    }
    let mut e = Scalar::decode_reduce(&ebuf);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            P = P * e;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
        e += e;
    }
    tt.sort();
    let x = P.encode().0[0];
    println!("Curve mul:       {:11.2}  ({})", (tt[4] as f64) / 1000.0, x.to_u64());
}

fn bench_curve_mulgen() {
    let mut ebuf = [0u8; 40];
    for i in 0..ebuf.len() {
        ebuf[i] = core_cycles() as u8;
    }
    let mut e = Scalar::decode_reduce(&ebuf);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let P = Point::mulgen(e);
            if (P.encode().0[0].to_u64() & 1) == 0 {
                e += e;
            }
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
        e += e;
    }
    tt.sort();
    let x = e.encode()[0];
    println!("Curve mulgen:    {:11.2}  ({})", (tt[4] as f64) / 1000.0, x);
}

fn bench_curve_verify_muladd_vartime() {
    let mut prng = PRNG(core_cycles() as u128);
    let mut QQ = [Point::NEUTRAL; 50];
    let mut RR = [Point::NEUTRAL; 50];
    let mut ss = [Scalar::ZERO; 50];
    let mut kk = [Scalar::ZERO; 50];
    let mut tt = [0; 100];
    for i in 0..100 {
        for j in 0..QQ.len() {
            let mut ebuf = [0u8; 40];
            let mut sbuf = [0u8; 40];
            let mut kbuf = [0u8; 40];
            prng.next(&mut ebuf);
            prng.next(&mut sbuf);
            prng.next(&mut kbuf);
            QQ[j] = Point::mulgen(Scalar::decode_reduce(&ebuf));
            ss[j] = Scalar::decode_reduce(&sbuf);
            kk[j] = Scalar::decode_reduce(&kbuf);
            RR[j] = Point::mulgen(ss[j]) + kk[j]*QQ[j];
            if (prng.next_u64() & 1) != 0 {
                RR[j] = RR[j].double();
            }
        }
        let begin = core_cycles();
        for j in 0..QQ.len() {
            if QQ[j].verify_muladd_vartime(ss[j], kk[j], RR[j]) {
                prng.0 = prng.0.wrapping_add(j as u128);
            }
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let x = prng.next_u64();
    println!("Curve verify:    {:11.2}  ({})", (tt[49] as f64) / (QQ.len() as f64), x);
}

fn main() {
    bench_curve_add();
    bench_curve_double();
    bench_curve_double_x5();
    bench_curve_mul();
    bench_curve_mulgen();
    bench_curve_verify_muladd_vartime();
}
