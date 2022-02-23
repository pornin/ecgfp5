use ecgfp5::field::{GFp, GFp5};
use core::arch::x86_64::{_mm_lfence, _rdtsc};

fn core_cycles() -> u64 {
    unsafe {
        _mm_lfence();
        _rdtsc()
    }
}

fn bench_gfp_add() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut y = x + GFp::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x += y;
            y += x;
            x += y;
            y += x;
            x += y;
            y += x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp add:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp_sub() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut y = x + GFp::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x -= y;
            y -= x;
            x -= y;
            y -= x;
            x -= y;
            y -= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp sub:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp_mul() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut y = x + GFp::ONE;
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x *= y;
            y *= x;
            x *= y;
            y *= x;
            x *= y;
            y *= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp mul:         {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp_invert() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp invert:      {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp_legendre() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
            x = GFp::from_u64_reduce(x.legendre().to_u64().wrapping_add(x.to_u64()));
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp legendre:    {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp_sqrt() {
    let mut x = GFp::from_u64_reduce(core_cycles());
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let (r0, c0) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r0.to_u64()).wrapping_add(c0));
            let (r1, c1) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r1.to_u64()).wrapping_add(c1));
            let (r2, c2) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r2.to_u64()).wrapping_add(c2));
            let (r3, c3) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r3.to_u64()).wrapping_add(c3));
            let (r4, c4) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r4.to_u64()).wrapping_add(c4));
            let (r5, c5) = x.sqrt(); x = GFp::from_u64_reduce(x.to_u64().wrapping_add(r5.to_u64()).wrapping_add(c5));
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp sqrt:        {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.to_u64());
}

fn bench_gfp5_add() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut y = GFp5([x4, x3, x2, x1, x0]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x += y;
            y += x;
            x += y;
            y += x;
            x += y;
            y += x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let z = (x.0[0] + x.0[1] + x.0[2] + x.0[3] + x.0[4]).to_u64();
    println!("GFp5 add:        {:11.2}  ({})", (tt[4] as f64) / 6000.0, z);
}

fn bench_gfp5_sub() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut y = GFp5([x4, x3, x2, x1, x0]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x -= y;
            y -= x;
            x -= y;
            y -= x;
            x -= y;
            y -= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let z = (x.0[0] + x.0[1] + x.0[2] + x.0[3] + x.0[4]).to_u64();
    println!("GFp5 sub:        {:11.2}  ({})", (tt[4] as f64) / 6000.0, z);
}

fn bench_gfp5_mul() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut y = GFp5([x4, x3, x2, x1, x0]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x *= y;
            y *= x;
            x *= y;
            y *= x;
            x *= y;
            y *= x;
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let z = (x.0[0] + x.0[1] + x.0[2] + x.0[3] + x.0[4]).to_u64();
    println!("GFp5 mul:        {:11.2}  ({})", (tt[4] as f64) / 6000.0, z);
}

fn bench_gfp5_square() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x = x.square();
            x = x.square();
            x = x.square();
            x = x.square();
            x = x.square();
            x = x.square();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    let z = (x.0[0] + x.0[1] + x.0[2] + x.0[3] + x.0[4]).to_u64();
    println!("GFp5 square:     {:11.2}  ({})", (tt[4] as f64) / 6000.0, z);
}

fn bench_gfp5_invert() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
            x = x.invert();
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp5 invert:     {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.0[0].to_u64());
}

fn bench_gfp5_legendre() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let v0 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v0.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v0.to_u64())), x.0[2], x.0[3], x.0[4]]);
            let v1 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v1.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v1.to_u64())), x.0[2], x.0[3], x.0[4]]);
            let v2 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v2.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v2.to_u64())), x.0[2], x.0[3], x.0[4]]);
            let v3 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v3.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v3.to_u64())), x.0[2], x.0[3], x.0[4]]);
            let v4 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v4.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v4.to_u64())), x.0[2], x.0[3], x.0[4]]);
            let v5 = x.legendre(); x = GFp5([GFp::from_u64_reduce(x.0[0].to_u64().wrapping_add(v5.to_u64())), GFp::from_u64_reduce(x.0[1].to_u64().wrapping_sub(v5.to_u64())), x.0[2], x.0[3], x.0[4]]);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp5 legendre:   {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.0[0].to_u64());
}

fn bench_gfp5_sqrt() {
    let x0 = GFp::from_u64_reduce(core_cycles());
    let x1 = GFp::from_u64_reduce(core_cycles());
    let x2 = GFp::from_u64_reduce(core_cycles());
    let x3 = GFp::from_u64_reduce(core_cycles());
    let x4 = GFp::from_u64_reduce(core_cycles());
    let mut x = GFp5([x0, x1, x2, x3, x4]);
    let mut tt = [0; 10];
    for i in 0..10 {
        let begin = core_cycles();
        for _ in 0..1000 {
            let (s0, c0) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s0.0[0].to_u64().wrapping_add(c0)), s0.0[1], s0.0[2], s0.0[3], s0.0[4]]);
            let (s1, c1) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s1.0[0].to_u64().wrapping_add(c1)), s1.0[1], s1.0[2], s1.0[3], s1.0[4]]);
            let (s2, c2) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s2.0[0].to_u64().wrapping_add(c2)), s2.0[1], s2.0[2], s2.0[3], s2.0[4]]);
            let (s3, c3) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s3.0[0].to_u64().wrapping_add(c3)), s3.0[1], s3.0[2], s3.0[3], s3.0[4]]);
            let (s4, c4) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s4.0[0].to_u64().wrapping_add(c4)), s4.0[1], s4.0[2], s4.0[3], s4.0[4]]);
            let (s5, c5) = x.sqrt(); x = GFp5([GFp::from_u64_reduce(s5.0[0].to_u64().wrapping_add(c5)), s5.0[1], s5.0[2], s5.0[3], s5.0[4]]);
        }
        let end = core_cycles();
        tt[i] = end.wrapping_sub(begin);
    }
    tt.sort();
    println!("GFp5 sqrt:       {:11.2}  ({})", (tt[4] as f64) / 6000.0, x.0[0].to_u64());
}

fn main() {
    bench_gfp_add();
    bench_gfp_sub();
    bench_gfp_mul();
    bench_gfp_invert();
    bench_gfp_legendre();
    bench_gfp_sqrt();

    bench_gfp5_add();
    bench_gfp5_sub();
    bench_gfp5_mul();
    bench_gfp5_square();
    bench_gfp5_invert();
    bench_gfp5_legendre();
    bench_gfp5_sqrt();
}
