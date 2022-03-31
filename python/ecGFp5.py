#! /usr/bin/env python3

# =========================================================================
# Test implementation of ecGFp5.
#
# This file simulates in-VM computations. The GFp object (GFpImpl class)
# implements operations in GF(p), and keeps track of cost by
# incrementing the operation count for each executed operation. The GFp5
# object (GFp5Impl class) implements GF(p^5) operations on top of GFp.
# The EcGFp5 object (EcGFp5Impl class) implements ecGFp5 operations
# using the techniques deemed appropriate for efficient in-VM
# computations.
#
# Implementation techniques emulated here are described in the
# ecGFp5 whitepaper ("In-VM Implementation").
#
# WARNING: use only for tests! None of this code is constant-time.

# =========================================================================
# GF(p) implementation.

class GFpImpl:
    def __init__(self):
        p = 2**64 - 2**32 + 1
        self.p = p
        self.zero = GFpImpl.Element(self, 0)
        self.one = GFpImpl.Element(self, 1)
        self.minusone = GFpImpl.Element(self, p - 1)
        self.onehalf = GFpImpl.Element(self, (p + 1) // 2)
        self.countOp = 0
        # Precomputed constants for sqrt().
        # gg[i] = g^(2^i) mod m, for g = 7^((m-1)//2^32)
        # (7 is a non-QR modulo m, g is a primitive 2^32-th root of 1)
        gg_src = [ 1753635133440165772,  4614640910117430873,
                   9123114210336311365, 16116352524544190054,
                   6414415596519834757,  1213594585890690845,
                  17096174751763063430,  5456943929260765144,
                   9713644485405565297, 16905767614792059275,
                   5416168637041100469, 17654865857378133588,
                   3511170319078647661, 18146160046829613826,
                   9306717745644682924, 12380578893860276750,
                   6115771955107415310, 17776499369601055404,
                  16207902636198568418,  1532612707718625687,
                  17492915097719143606,   455906449640507599,
                  11353340290879379826,  1803076106186727246,
                  13797081185216407910, 17870292113338400769,
                          549755813888,       70368744161280,
                  17293822564807737345, 18446744069397807105,
                       281474976710656, 18446744069414584320 ]
        self.gg = []
        for i in range(0, 32):
            self.gg.append(GFpImpl.Element(self, gg_src[i]))

    def __call__(self, x):
        """
        Make a field element. If x is already an element of this field,
        then it is returned as is. Otherwise, x is converted to an integer,
        which is reduced modulo the ring modulus, and used to make a new
        value.
        """
        if isinstance(x, GFpImpl.Element):
            return x
        return GFpImpl.Element(self, int(x) % self.p)

    def Decode(self, bb):
        """
        Decode an element from 8 bytes (unsigned little-endian convention).
        An eception is thrown if the value is not in the 0..p-1 range.
        """
        if len(bb) != 8:
            raise Exception('Invalid encoded value (wrong length = {0})'.format(len(bb)))
        x = int.from_bytes(bb, byteorder='little')
        if x >= self.p:
            raise Exception('Invalid encoded value (not lower than modulus)')
        return GFpImpl.Element(self, x)

    def DecodeReduce(self, bb):
        """
        Decode an element from bytes. All provided bytes are read, in
        unsigned little-endian convention; the value is then reduced
        modulo the ring modulus.
        """
        x = int.from_bytes(bb, byteorder='little')
        return GFpImpl.Element(self, x % self.p)

    class Element:
        def __init__(self, ring, value):
            self.ring = ring
            self.x = int(value)

        def __getattr__(self, name):
            if name == 'modulus':
                return self.ring.p
            else:
                raise AttributeError()

        def __int__(self):
            """
            Conversion to an integer returns the value in the 0..p-1 range.
            """
            return self.x

        def valueOfOther(self, b):
            if isinstance(b, GFpImpl.Element):
                if self.ring is b.ring:
                    return b.x
                if self.ring.p != b.ring.p:
                    raise Exception('ring mismatch')
                return b.x
            elif isinstance(b, int):
                return b % self.ring.p
            else:
                return False

        def __add__(self, b):
            # opcode: add
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(self.x + b)

        def __radd__(self, b):
            # opcode: add
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(b + self.x)

        def __sub__(self, b):
            # opcode: sub
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(self.x - b)

        def __rsub__(self, b):
            # opcode: sub
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(b - self.x)

        def __neg__(self):
            # opcode: neg
            self.ring.countOp += 1
            return self.ring(-self.x)

        def __mul__(self, b):
            # opcode: mul
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(self.x * b)

        def __rmul__(self, b):
            # opcode: mul
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            return self.ring(b * self.x)

        def square(self):
            # opcode: mul
            self.ring.countOp += 1
            return self.ring(self.x * self.x)

        def half(self):
            # opcode: mul
            return self * self.ring.onehalf

        def asbool(self):
            if self.x == 0:
                return False
            if self.x == 1:
                return True
            raise Exception('not a 0/1 value')

        def bvOfOther(self, b):
            if isinstance(b, GFpImpl.Element):
                if b.ring is self.ring:
                    return b
                return self.ring(b.x)
            if isinstance(b, int):
                return self.ring(b)
            return False

        def bv_and(self, b):
            # opcode: and
            b = self.bvOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            v1 = self.asbool()
            v2 = b.asbool()
            if v1 and v2:
                return self.ring.one
            else:
                return self.ring.zero

        def bv_or(self, b):
            # opcode: or
            b = self.bvOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            v1 = self.asbool()
            v2 = b.asbool()
            if v1 or v2:
                return self.ring.one
            else:
                return self.ring.zero

        def bv_xor(self, b):
            # opcode: xor
            b = self.bvOfOther(b)
            if b is False:
                return NotImplemented
            self.ring.countOp += 1
            v1 = self.asbool()
            v2 = b.asbool()
            if v1 == v2:
                return self.ring.zero
            else:
                return self.ring.one

        def bv_not(self):
            # opcode: not
            v = self.asbool()
            self.ring.countOp += 1
            if v:
                return self.ring.zero
            else:
                return self.ring.one

        def __truediv__(self, y):
            # opcode: div

            # We explicitly check for division by 0; otherwise, the
            # implementation would return 0 (returning zero on a zero
            # divisor is a convenient behaviour, but not the one of the
            # VM we are emulating).
            #
            # We use a binary GCD. Invariants:
            #   a*x = u*y mod p
            #   b*x = v*y mod p
            # The GCD ends with b = 1, in which case v = x/y mod p.
            a = self.valueOfOther(y)
            if a is False:
                return NotImplemented
            if a == 0:
                raise Exception('division by zero')
            m = self.ring.p
            b = m
            u = self.x
            v = 0
            while a != 0:
                if (a & 1) == 0:
                    a >>= 1
                    if (u & 1) != 0:
                        u += m
                    u >>= 1
                else:
                    if a < b:
                        a, b = b, a
                        u, v = v, u
                    a -= b
                    if u < v:
                        u += m
                    u -= v
            # Note: if the divisor is zero, then we immediately arrive
            # here with v = 0, which is what we want.
            self.ring.countOp += 1
            return self.ring(v)

        def __rtruediv__(self, y):
            return self.ring(y).__truediv__(self)

        def __floordiv__(self, y):
            return self.__truediv__(y)

        def __rfloordiv__(self, y):
            return self.ring(y).__truediv__(self)

        # __eq__ is for the '==' operator and returns a Boolean.
        # See equals() for the variant that returns 0 or 1 in the ring.
        def __eq__(self, b):
            # opcode: eq
            if isinstance(b, GFpImpl.Element):
                if self.ring.p != b.ring.p:
                    return False
                self.ring.countOp += 1
                return self.x == b.x
            else:
                self.ring.countOp += 1
                return self.x == int(b)

        # __ne__ is for the '!=' operator and returns a Boolean.
        # See notequals() for the variant that returns 0 or 1 in the ring.
        def __ne__(self, b):
            # opcode: neq
            if isinstance(b, GFpImpl.Element):
                if self.ring.p != b.ring.p:
                    return True
                self.ring.countOp += 1
                return self.x != b.x
            else:
                self.ring.countOp += 1
                return self.x != int(b)

        def equals(self, b):
            if self == b:
                return self.ring.one
            else:
                return self.ring.zero

        def notequals(self, b):
            if self != b:
                return self.ring.one
            else:
                return self.ring.zero

        def iszero(self):
            return self.equals(self.ring.zero)

        def isone(self):
            return self.equals(self.ring.one)

        def isminusone(self):
            return self.equals(self.ring.minusone)

        def __repr__(self):
            return self.x.__repr__()

        def __str__(self):
            return self.x.__str__()

        def __format__(self, fspec):
            return self.x.__format__(fspec)

        def __bytes__(self):
            return self.x.to_bytes(self.ring.encodedLen, byteorder='little')

        def sqrt(self):
            """
            Compute a square root of the current value. This function
            returns two ring elements (s, c):
              - If the value is a square, s is a square root of this
                value, and c is 1.
              - Otherwise, the value is not a square, s is 0 and c is 0.
            Which square root is obtained is undetermined.
            """
            # We use Tonelli-Shanks without testing for intermediate
            # values to emulate a static circuit (no data-dependent
            # branching).
            #   Input: x
            #   Output: sqrt(x), or False if no square root exists
            #   Definitions:
            #      n = 32
            #      q = 2^32 - 1
            #      p = 2^64 - 2^32 + 1 = q*2^n + 1  (modulus)
            #      g = 7^q mod p  (primitive 2^n root of 1 in GF(p))
            #      g[j] = g^(2^j)  (j = 0 to n-1) (precomputed in ring.gg)
            #   Init:
            #      u <- x^((q+1)/2)
            #      v <- x^q
            #      invariant: u^2 = x*v
            #   Process:
            #      for i = n-1 down to 1:
            #          w = v^(2^(i-1))  (i-1 squarings)
            #          if w == -1 then:
            #              v <- v*g[n-i]
            #              u <- u*g[n-i-1]
            #      if v == 0 or 1 then:
            #          return (u, 1)   (square root found)
            #      else:
            #          return (0, 0)   (no square root)
            #
            # The algorithm works for the following reasons:
            #
            #   g[n-i] is a primitive 2^i-th root of 1, thus:
            #      g[n-i]^(2^i) = 1
            #      g[n-i]^(2^(i-1)) = -1
            #   Moreover, g[n-i-1] is a square root of g[n-i].
            #
            #   We first assume that x is a quadratic residue.
            #   When entering each loop iteration, we have:
            #       v^(2^i) = 1
            #   This is true initially, because:
            #       (x^q)^(2^(n-1)) = x^((q-1)/2) = 1 (since x is a QR)
            #   If v is a 2^i-th root of 1, then v = g[n-i]^t for
            #   some t in [0..2^(i-1)] (since g[n-i] is a generator of
            #   the group of 2^i-th roots of 1).
            #   If t is even, then:
            #       w = v^(2^(i-1))
            #         = g[n-i]^(t*2^(i-1))
            #         = (g[n-i]^(2^i))^(t/2)
            #         = 1
            #   In that case, v is already ready for the next iteration
            #   (it already fulfills v^(2^(i-1)) = 1).
            #   Otherwise, t is odd, and then:
            #       w = v^(2^(i-1))
            #         = g[n-i]^((2*k+1)*2^(i-1))   (with k = (t-1)/2)
            #         = (g[n-i]^(2^(i-1)))*((g[n-i]^(2^i))^k)
            #         = -1
            #   Then, by multiplying v by g[n-i], we get:
            #       (v*g[n-i])^(2^(i-1)) = v^(2^(i-1)) * g[n-i]^(2^(i-1)
            #                            = -1 * -1 = 1
            #   This shows that for all loop iterations, we indeed have
            #   v^(2^i) = 1 on entry, and v^(2^(i-1)) = 1 on exit (as
            #   long as, indeed, x is a quadratic residue). In particular,
            #   after the last iteration, we have v = 1.
            #
            #   Note that the invariant u^2 = x*v is maintained at all
            #   times, since whenever we multiply v by g[n-i], we
            #   also multiply u by g[n-i-1], which is a square root of
            #   g[n-i]. This is true regardless of whether x is a QR
            #   or not. Therefore, if v = 1 after the loop, then it
            #   MUST be that u^2 = x.
            #
            #   If u = 0 on input, then all u and v are zero throughout,
            #   and the algorithm returns u = 0, which is correct.
            #   If u is not a QR, then none of the values w is equal
            #   to -1, and v is unchanged throughout the algorithm (and
            #   thus not equal to 0 or 1 at the end).

            #      u <- x^((q+1)/2)
            #      v <- x^q
            x = self
            u = x.msquare(31)
            v = u.square() / (x + x.iszero())  # +iszero() to avoid div by 0
            g = self.ring.gg
            n = 32
            for j in range(1, n):
                i = n - j
                w = v.msquare(i - 1)
                cc = w.isminusone()
                v = cc.select(v, v * g[n-i])
                u = cc.select(u, u * g[n-i-1])
            cc = v.iszero().bv_or(v.isone())
            return (u * cc, cc)

        def legendre(self):
            """
            Raise this value to the power (p-1)/2 (Legendre symbol).
            """
            # a^((p-1)/2) = a^(2^63 - 2^31) = a^(2^63) / a^(2^31)
            # We use the fact that division is fast in our compute
            # model. We still need a special trick to avoid division
            # by zero (result is still correctly zero in that case).
            v1 = self.msquare(31)
            v2 = v1.msquare(32)
            v1 += v1.iszero()
            return v2 / v1

        def msquare(self, n):
            """
            Return this value raised to the power 2^n (i.e. n successive
            squarings).
            """
            a = self
            for i in range(0, n):
                a = a.square()
            return a

        def invert(self):
            """
            Wrapper for inversion (uses the division implementation).
            """
            return 1 / self

        def select(self, a0, a1):
            """
            Return a0 if this value is 0, a1 if this value is 1.
            If this value is not 0 or 1 then an exception is thrown.
            """
            # opcode: cdrop

            # This emulates a conditional drop (cdrop).
            # Cost is 1 if a0 and a1 are elements of GF(p), 5 if they
            # are elements of GF(p^5).
            if isinstance(a0, GFpImpl.Element) and isinstance(a1, GFpImpl.Element):
                cost = 1
            elif isinstance(a0, GFp5Impl.Element) and isinstance(a1, GFp5Impl.Element):
                cost = 5
            else:
                raise Exception('cannot select: operand mismatch')
            v = self.asbool()
            self.ring.countOp += cost
            if v:
                return a1
            else:
                return a0

        def check1(self):
            if self.x > 1:
                raise Exception('1-bit value is out of range')

        def check32(self):
            if self.x >= 2**32:
                raise Exception('32-bit value is out of range')

        def add32(self, b):
            # opcode: u32add.unsafe
            self.check32()
            b.check32()
            self.ring.countOp += 1
            v = self.x + b.x
            if v >= 2**32:
                return (self.ring(v - 2**32), self.ring.one)
            else:
                return (self.ring(v), self.ring.zero)

        def addc32(self, b, c):
            # opcode: u32addc.unsafe
            self.check32()
            b.check32()
            c.check1()
            self.ring.countOp += 1
            v = self.x + b.x + c.x
            if v >= 2**32:
                return (self.ring(v - 2**32), self.ring.one)
            else:
                return (self.ring(v), self.ring.zero)

        def and32(self, b):
            # opcode: u32and
            self.check32()
            b.check32()
            self.ring.countOp += 1
            return self.ring(self.x & b.x)

        def shl32(self, n):
            # opcode: u32shl
            # This could use a mul with a constant (we don't leverage the
            # 'truncate to 32 bits' behaviour), in case u32shl has cost
            # more than 1 in the VM.
            self.check32()
            self.ring.countOp += 1
            return self.ring(self.x << n)

        def shr32(self, n):
            # opcode: u32shr
            self.check32()
            self.ring.countOp += 1
            return self.ring(self.x >> n)

        def gte32(self, b):
            # opcode: u32gte.unsafe
            self.check32()
            b.check32()
            self.ring.countOp += 1
            if self.x >= b.x:
                return self.ring.one
            else:
                return self.ring.zero

GFp = GFpImpl()

# =========================================================================

class GFp5Impl:
    def __init__(self):
        self.zero = GFp5Impl.Element(self,
            GFp.zero, GFp.zero, GFp.zero, GFp.zero, GFp.zero)
        self.one = GFp5Impl.Element(self,
            GFp.one, GFp.zero, GFp.zero, GFp.zero, GFp.zero)

    def __call__(self, x):
        if isinstance(x, GFp5Impl.Element):
            return x
        if (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 5:
            return GFp5Impl.Element(self,
                GFp(x[0]), GFp(x[1]), GFp(x[2]), GFp(x[3]), GFp(x[4]))
        return GFp5Impl.Element(self, GFp(x),
            GFp.zero, GFp.zero, GFp.zero, GFp.zero)

    class Element:
        def __init__(self, field, a0, a1, a2, a3, a4):
            self.field = field
            self.a0 = a0
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
            self.a4 = a4

        def __getattr__(self, name):
            raise AttributeError()

        def valueOfOther(self, b):
            if isinstance(b, GFp5Impl.Element):
                return b
            elif isinstance(b, GFpImpl):
                return GFp5Impl.Element(self.field,
                    b, GFp.zero, GFp.zero, GFp.zero, GFp.zero)
            elif isinstance(b, int):
                return GFp5Impl.Element(self.field,
                        GFp(b), GFp.zero, GFp.zero, GFp.zero, GFp.zero)
            else:
                return False

        def __add__(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            return GFp5Impl.Element(self.field,
                self.a0 + b.a0,
                self.a1 + b.a1,
                self.a2 + b.a2,
                self.a3 + b.a3,
                self.a4 + b.a4)

        def __radd__(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            return GFp5Impl.Element(self.field,
                b.a0 + self.a0,
                b.a1 + self.a1,
                b.a2 + self.a2,
                b.a3 + self.a3,
                b.a4 + self.a4)

        def __sub__(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            return GFp5Impl.Element(self.field,
                self.a0 - b.a0,
                self.a1 - b.a1,
                self.a2 - b.a2,
                self.a3 - b.a3,
                self.a4 - b.a4)

        def __rsub__(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            return GFp5Impl.Element(self.field,
                b.a0 - self.a0,
                b.a1 - self.a1,
                b.a2 - self.a2,
                b.a3 - self.a3,
                b.a4 - self.a4)

        def addk01(self, b0, b1):
            # Adding an element of GF(p^5) known to be b0+b1*z,
            # i.e. only two non-zero coefficients.
            return GFp5Impl.Element(self.field,
                self.a0 + b0, self.a1 + b1, self.a2, self.a3, self.a4)

        def subk1(self, b1):
            # Subtracting an element of GF(p^5) known to be b1*z,
            # i.e. only one non-zero coefficient.
            return GFp5Impl.Element(self.field,
                self.a0, self.a1 - b1, self.a2, self.a3, self.a4)

        def __neg__(self):
            return GFp5Impl.Element(self.field,
                -self.a0,
                -self.a1,
                -self.a2,
                -self.a3,
                -self.a4)

        def mulconstgf(self, b):
            # Multiplication by a value in GF(p).
            return GFp5Impl.Element(self.field,
                self.a0 * b, self.a1 * b, self.a2 * b, self.a3 * b, self.a4 * b)

        def __mul__(self, b):
            if isinstance(b, GFpImpl.Element):
                return self.mulconstgf(b)
            elif isinstance(b, int):
                return self.mulconstgf(GFp(b))
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            c0 = self.a0 * b.a0 + 3 * (self.a1 * b.a4 + self.a2 * b.a3 + self.a3 * b.a2 + self.a4 * b.a1)
            c1 = self.a0 * b.a1 + self.a1 * b.a0 + 3 * (self.a2 * b.a4 + self.a3 * b.a3 + self.a4 * b.a2)
            c2 = self.a0 * b.a2 + self.a1 * b.a1 + self.a2 * b.a0 + 3 * (self.a3 * b.a4 + self.a4 * b.a3)
            c3 = self.a0 * b.a3 + self.a1 * b.a2 + self.a2 * b.a1 + self.a3 * b.a0 + 3 * (self.a4 * b.a4)
            c4 = self.a0 * b.a4 + self.a1 * b.a3 + self.a2 * b.a2 + self.a3 * b.a1 + self.a4 * b.a0
            return GFp5Impl.Element(self.field, c0, c1, c2, c3, c4)

        def __rmul__(self, b):
            if isinstance(b, GFpImpl.Element):
                return self.mulconstgf(b)
            elif isinstance(b, int):
                return self.mulconstgf(GFp(b))
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            c0 = b.a0 * self.a0 + 3 * (b.a1 * self.a4 + b.a2 * self.a3 + b.a3 * self.a2 + b.a4 * self.a1)
            c1 = b.a0 * self.a1 + b.a1 * self.a0 + 3 * (b.a2 * self.a4 + b.a3 * self.a3 + b.a4 * self.a2)
            c2 = b.a0 * self.a2 + b.a1 * self.a1 + b.a2 * self.a0 + 3 * (b.a3 * self.a4 + b.a4 * self.a3)
            c3 = b.a0 * self.a3 + b.a1 * self.a2 + b.a2 * self.a1 + b.a3 * self.a0 + 3 * (b.a4 * self.a4)
            c4 = b.a0 * self.a4 + b.a1 * self.a3 + b.a2 * self.a2 + b.a3 * self.a1 + b.a4 * self.a0
            return GFp5Impl.Element(self.field, c0, c1, c2, c3, c4)

        def half(self):
            return self.mulconstgf(GFp.onehalf)

        def square(self):
            c0 = self.a0 * self.a0 + 6 * (self.a1 * self.a4 + self.a2 * self.a3)
            c1 = 2 * self.a0 * self.a1 + 3 * (self.a3 * self.a3 + 2 * self.a2 * self.a4)
            c2 = self.a1 * self.a1 + 2 * self.a0 * self.a2 + 6 * self.a3 * self.a4
            c3 = 2 * (self.a0 * self.a3 + self.a1 * self.a2) + 3 * self.a4 * self.a4
            c4 = self.a2 * self.a2 + 2 * (self.a0 * self.a4 + self.a1 * self.a3)
            return GFp5Impl.Element(self.field, c0, c1, c2, c3, c4)

        def msquare(self, n):
            a = self
            for i in range(0, n):
                a = a.square()
            return a

        def frob1(self):
            """
            Apply the Frobenius operator once (i.e. raise this value to the
            power p).
            """
            # Since z^5 = 3 in the field, and p = 1 mod 5, we have:
            #   (z^i)^p = 3^(i*floor(p/5))*z^i
            # The Frobenius operator is a field automorphism, so we just
            # have to multiply the coefficients by the right values.
            c0 = self.a0
            c1 = self.a1 * GFp(1041288259238279555)   # 3^(floor(p/5))
            c2 = self.a2 * GFp(15820824984080659046)  # 3^(2*floor(p/5))
            c3 = self.a3 * GFp(211587555138949697)    # 3^(3*floor(p/5))
            c4 = self.a4 * GFp(1373043270956696022)   # 3^(4*floor(p/5))
            return GFp5Impl.Element(self.field, c0, c1, c2, c3, c4)

        def frob2(self):
            """
            Apply the Frobenius operator twice (i.e. raise this value to the
            power p^2).
            """
            c0 = self.a0
            c1 = self.a1 * GFp(15820824984080659046)  # 9^(floor(p/5))
            c2 = self.a2 * GFp(1373043270956696022)   # 9^(2*floor(p/5))
            c3 = self.a3 * GFp(1041288259238279555)   # 9^(3*floor(p/5))
            c4 = self.a4 * GFp(211587555138949697)    # 9^(4*floor(p/5))
            return GFp5Impl.Element(self.field, c0, c1, c2, c3, c4)

        def invert(self):
            """
            Invert this element in the field. If this value is 0 then this
            returns 0.
            """
            t0 = self.frob1()         # t0 <- a^p
            t1 = t0 * t0.frob1()      # t1 <- a^(p + p^2)
            t2 = t1 * t1.frob2()      # t2 <- a^(p + p^2 + p^3 + p^4)
            # Let r = 1 + p + p^2 + p^3 + p^4. We have a^r = a * t2. Also,
            # (a^r)^(p-1) = a^(p^5-1) = 1, so a^r is in GF(p) (b^(p-1) = 1 for
            # all non-zero elements in GF(p), and that's p-1 solutions to a
            # polynomial of degree p-1, so it works in the other direction too:
            # all values b such that b^(p-1) = 1 must be in GF(p)). Thus,
            # We can compute a^r as only the low coefficient of a*t2 (into t3).
            t3 = self.a0 * t2.a0 + 3 * (self.a1 * t2.a4 + self.a2 * t2.a3 + self.a3 * t2.a2 + self.a4 * t2.a1)
            # 1/a = a^(r-1) / a^r = t2 / t3. We just need to invert t3 (which
            # is in GF(p)) and multiply t2 by it.
            # If input 'a' is zero then we will divide 0 by 0, which is not
            # defined; we need a small corrective step to make divisor t3
            # equal to 1 in that case (the final output will still be zero,
            # since in such a case t2 = (0,0,0,0,0)).
            t3 += t3.iszero()
            t4 = t3.invert()
            return GFp5Impl.Element(self.field,
                t4 * t2.a0, t4 * t2.a1, t4 * t2.a2, t4 * t2.a3, t4 * t2.a4)

        def __truediv__(self, y):
            b = self.valueOfOther(y)
            if b is False:
                return NotImplemented
            return self * b.invert()

        def __rtruediv__(self, y):
            return y.__truediv__(self)

        def __floordiv__(self, y):
            return self.__truediv__(y)

        def __rfloordiv__(self, y):
            return y.__truediv__(self)

        # __eq__ is for the '==' operator and returns a Boolean.
        # See equals() for the variant that returns 0 or 1 in the ring.
        def __eq__(self, b):
            return self.equals(b).asbool()

        def equals(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            c0 = self.a0.equals(b.a0)
            c1 = self.a1.equals(b.a1)
            c2 = self.a2.equals(b.a2)
            c3 = self.a3.equals(b.a3)
            c4 = self.a4.equals(b.a4)
            return c0.bv_and(c1).bv_and(c2).bv_and(c3).bv_and(c4)

        # __ne__ is for the '!=' operator and returns a Boolean.
        # See notequals() for the variant that returns 0 or 1 in the ring.
        def __ne__(self, b):
            return self.notequals(b).asbool()

        def notequals(self, b):
            b = self.valueOfOther(b)
            if b is False:
                return NotImplemented
            c0 = self.a0.notequals(b.a0)
            c1 = self.a1.notequals(b.a1)
            c2 = self.a2.notequals(b.a2)
            c3 = self.a3.notequals(b.a3)
            c4 = self.a4.notequals(b.a4)
            return c0.bv_or(c1).bv_or(c2).bv_or(c3).bv_or(c4)

        def __repr__(self):
            return "GFp5(%s,%s,%s,%s,%s)" % (self.a0, self.a1, self.a2, self.a3, self.a4)

        def __str__(self):
            return "(%s,%s,%s,%s,%s)" % (self.a0, self.a1, self.a2, self.a3, self.a4)

        def __bytes__(self):
            return self.a0.__bytes__() + self.a1.__bytes__() + self.a2.__bytes__() + self.a3.__bytes__() + self.a4.__bytes__()

        def iszero(self):
            c0 = self.a0.iszero()
            c1 = self.a1.iszero()
            c2 = self.a2.iszero()
            c3 = self.a3.iszero()
            c4 = self.a4.iszero()
            return c0.bv_and(c1).bv_and(c2).bv_and(c3).bv_and(c4)

        def isone(self):
            c0 = self.a0.isone()
            c1 = self.a1.iszero()
            c2 = self.a2.iszero()
            c3 = self.a3.iszero()
            c4 = self.a4.iszero()
            return c0.bv_and(c1).bv_and(c2).bv_and(c3).bv_and(c4)

        def isminusone(self):
            c0 = self.a0.isminusone()
            c1 = self.a1.iszero()
            c2 = self.a2.iszero()
            c3 = self.a3.iszero()
            c4 = self.a4.iszero()
            return c0.bv_and(c1).bv_and(c2).bv_and(c3).bv_and(c4)

        def legendre(self):
            """
            Legendre symbol:
                0 for zero
                1 for a non-zero quadratic residue
               -1 for a non-quadratic residue
            """
            # Let r = 1 + p + p^2 + p^3 + p^4 (as in invert()).
            # We have a^((p^5-1)/2) = a^((p-1)*r/2) = (a^r)^((p-1)/2).
            # a^r is in GF(p), thus a is a square in GF(p^5) if and only
            # if a^r is a square in GF(p). We compute a^r with a few
            # Frobenius operators and multiplications, then apply Legendre's
            # symbol on a^r (with an exponentiation, which is faster than
            # the Jacobi symbol in our compute model).
            t0 = self.frob1()         # t0 <- a^p
            t1 = t0 * t0.frob1()      # t1 <- a^(p + p^2)
            t2 = t1 * t1.frob2()      # t2 <- a^(p + p^2 + p^3 + p^4)
            t3 = self.a0 * t2.a0 + 3 * (self.a1 * t2.a4 + self.a2 * t2.a3 + self.a3 * t2.a2 + self.a4 * t2.a1)
            # t3 contains a^r; its Legendre symbol is equal to that of the
            # original input.
            return t3.legendre()

        def sqrt(self):
            """
            Returns (r, c). If this value is a square, then c = 1 and r is a
            square root of this value; otherwise, r = 0 and c = 0.
            """
            # a = (a^r)/(a^(r-1)), thus sqrt(a) = sqrt(a^r) / a^((r-1)/2).
            # a^r is in GF(p). We have:
            #   (r-1)/2 = (p + p^2 + p^3 + p^4)/2
            #           = p*(1 + p^2)*((p + 1)/2)
            # Thus:
            #   d <- a^((p+1)/2)
            #   e <- frob1(d * frob2(d))
            # This computes a^((r-1)/2) in e. We then compute a^r as:
            #   a^r = a * e^2
            # Since a^r is in GF(p), the multiplication can be optimized
            # (we compute only the low coefficient).
            #
            # (p+1)/2 = 2^63 - 2^31 + 1. Since we have efficient divisions, we
            # compute a^((p+1)/2) = a * a^(2^63) / a^(2^31). Note that the
            # division routine in GF(p^5) returns 0 when the divisor is 0,
            # so we do not have to make a correction here.
            v = self.msquare(31)
            d = (self * v.msquare(32)) / v  # GFp5.invert() tolerates 0
            e = (d * d.frob2()).frob1()
            f = e.square()
            g = self.a0 * f.a0 + 3 * (self.a1 * f.a4 + self.a2 * f.a3 + self.a3 * f.a2 + self.a4 * f.a1)
            # Now e contains a^((r-1)/2) and g contains a^r. The square root in
            # GF(p) returns (0, 0) on a non-square, so we can use the value
            # directly.
            (s, c) = g.sqrt()
            e = e.invert()
            return (e * s, c)

GFp5 = GFp5Impl()

# =========================================================================

# Curve: a double-odd curve E with equation:
#     y^2 = x*(x^2 + a*x + b)
# Constants:
#     a = 2
#     b = 263*z
# The curve contains 2*n points, with n prime:
# n = 1067993516717146951041484916571792702745057740581727230159139685185762082554198619328292418486241
#
# We work with the group: G = { P+N | P \in E, n*P = inf }
# i.e. the set of points P+N where N = (0,0) (the unique point of order 2
# on the curve) and P is in the subgroup of points of order n in the curve.
# The group law is:
#    law(P1+N, P2+N) = P1+P2+N
# The neutral is N. This makes a prime-order group; all the elements
# (including N itself) have defined x and y coordinates, which is why we
# work on that group.
#
# An element of the group can be uniquely encoded into a field element w:
#    N = (0,0) is encoded as w = 0
#    P+N = (x,y) != N is encoded as w = y/x
# Decoding works as follows:
#    if w == 0:
#        return N
#    else:
#        Solve the equation x^2 + (a - w^2)*x + b = 0:
#            delta <- (a - w^2)^2 - 4*b
#            if delta is not a QR:
#                return Failed (w is not valid)
#            x1 <- (w^2 - a + sqrt(delta))/2
#            x2 <- (w^2 - a - sqrt(delta))/2
#            If x1 is a QR:
#                x <- x2
#            else:
#                x <- x1
#            return (x,w*x)
#            
# The encoding/decoding process is unambiguous; encoding is injective (two
# distinct elements of G yield two distinct values of w), and for any
# element P+N of G, only a single w in the field can be decoded into that
# point.
#
#
# Inside the implementation, we work with an isomorphic curve in short
# Weierstraß form. The Weierstraß curve Ew is:
#     Y^2 = X^3 + A*X + B
# Constants:
#     A = (3*b - a^2)/3 = 6148914689804861439 + 263*z 
#     B = a*(2*a^2 - 9*b)/27 = 15713893096167979237 + 6148914689804861265*z
# We map G into Ew[n] (the subgroup of order n of the curve):
#     psi :     G --> Ew[n]
#           (x,y) |-> point-at-infinity if (x,y) = (0,0)
#                     (X,Y) = (b/x + a/3, -b*y/x^2)
# It is convenient to combine the decoding process and the map into a
# single operation. Indeed, the map is really the combination of two
# transforms:
#  1. Add N to the decoded point on curve E: (x,y) -> (x',y') = (b/x, -b*y/x^2)
#  2. Isomorphism of curves E -> Ew: (X,Y) = (x' + a/3, y')
# The decoding process can do the first part ("add N") for free, by
# modifying the decoding process above in two places:
#   - Instead of choosing the solution x which is not a QR, choose the
#     other one.
#   - Compute y as y = -w*x.
# This is due to the following fact: the value w is really the slope of the
# line from N = (0,0) to the point P+N = (x,y); by the definition of the
# point addition rule, the same line (hence with the same slope) also
# intersects the curve on a third point which is -P. By choosing the point
# whose x coordinate is a square, we select this '-P' instead of 'P+N'.
#
# The encoding process from the (X,Y) (Weierstraß) coordinates is then:
#     The point-at-infinity is encoded into 0.
#     For non-infinity point (X,Y):
#         (x',y') <- (X - a/3, Y)
#         (x,y) <- (b/x', -b*y'/x'^2)
#         w <- y/x
#     which can be simplified into:
#         w <- Y / (a/3 - X)

class EcGFp5Impl:

    def __init__(self):
        self.a = GFp5((2, 0, 0, 0, 0))
        self.b = GFp5((0, 263, 0, 0, 0))
        self.bmul4_1 = 4 * self.b.a1
        self.adiv3 = self.a / 3
        self.A = (3*self.b - self.a.square()) / 3
        self.A0 = self.A.a0
        self.A1 = self.A.a1
        self.B = self.a*(2*self.a.square() - 9*self.b) / 27
        self.B0 = self.B.a0
        self.B1 = self.B.a1
        self.neutral = EcGFp5Impl.Point(self, GFp5.zero, GFp5.zero, GFp.one)
        # subgroup order n, in 32-bit limbs, little-endian order.
        self.n = [ GFp(0x948BFFE1), GFp(0xE80FD996), GFp(0xD724A09C),
                   GFp(0xE8885C39), GFp(0xCFB80639), GFp(0x7FFFFFE6),
                   GFp(0x00000016), GFp(0x7FFFFFF1), GFp(0x80000007),
                   GFp(0x7FFFFFFD) ]
        # 2^320-n, in 32-bit limbs, little-endian order.
        self.minusn = [ GFp(0xF3FF2384), GFp(0x888B2365), GFp(0x709AFCFC),
                        GFp(0x47BF9D99), GFp(0x3047F9D3), GFp(0x0000000D),
                        GFp(0x7FFFFFF3), GFp(0x80000009), GFp(0xFFFFFFFA),
                        GFp(0x80000002) ]
        # Preferred window size (in bits). Valid values are in 2..10.
        # Optimal value is likely to be 4 or 5, depending on the
        # individual costs of operations in the field and on the curve.
        self.window_width = 4

    def decode(self, w):
        """
        Returned value: (P, c)
        On decoding success: P = point, c = 1
        On decoding failure: P = infinity, c = 0
        Note: if input is w = 0, then decoding succeeds and
        returns P = infinity, c = 1
        """
        e = w.square() - self.a
        delta = e.square().subk1(self.bmul4_1)
        (r, c) = delta.sqrt()
        x1 = (e + r).half()
        x2 = (e - r).half()
        x = x1.legendre().isone().select(x2, x1)
        y = -w*x
        inf = c.bv_not()         # Always get infinity on failure to decode
        c = c.bv_or(w.iszero())  # ... but if w = 0, this is not a real failure
        X = x + self.adiv3
        Y = y
        return (EcGFp5Impl.Point(self, X, Y, inf), c)

    def validate(self, w):
        """
        Verify that w could be decoded successfully (i.e. decode()
        would return (P,1) for some point P). Returned value is 1
        on success, 0 on error. Note that if w is zero, then 1 is
        returned.
        """
        e = w.square() - self.a
        delta = e.square().subk1(self.bmul4_1)
        return delta.legendre().equals(GFp.one).bv_or(w.iszero())

    def int_to_limbs(self, e):
        # Convenience function for tests: convert a source integer
        # (a Python big integer) into the expected in-VM format (a
        # sequence of 32-bit limbs, each held in a GF(p) element).
        # Limbs are in little-endian order. Size is fixed (10 limbs);
        # if the integer is negative, or not lower than 2^320, then
        # an exception is thrown.
        if e < 0:
            raise Exception('integer is negative')
        v = []
        for i in range(0, 10):
            v.append(GFp(e & 0xFFFFFFFF))
            e = e >> 32
        if e != 0:
            raise Exception('integer is too large')
        return v

    def check_scalar(self, v):
        """
        Input v is assumed to be an array of exactly ten GF(p) elements,
        each in the 0..2^32-1 range, encoding a scalar in unsigned
        little-endian order. A scalar is an integer modulo n (the prime
        subgroup order). This function checks that the scalar is indeed
        strictly lower than n. Returned value is 1 if the scalar is in
        the proper range, 0 otherwise.
        """
        # We add 2^320-n to the scalar; we do not keep the result, but
        # we look at the final carry: that carry will be 0 if and only if
        # the scalar was in the proper range.
        (_, c) = v[0].add32(self.minusn[0])
        for i in range(1, 10):
            (_, c) = v[i].addc32(self.minusn[i], c)
        return c.bv_not()

    def scalar_to_signed_padded(self, v, w):
        """
        Given a scalar v (ten 32-bit limbs in little-endian order,
        value in the 0..n-1 range), return two sequences t[] and s[]
        containing elements of GF(p), with the following characteristics:
          - len(s) = len(t) = ceil(321/w)
          - for all i: 0 <= t[i] <= 2^(w-1)
          - for all i: s[i] = 0 or 1
          - v = Sum_i t[i]*(1-2*s[i])*2^(w*i) mod n
          - t[len(t)-1] >= 1 and s[len(t)-1] = 0
        The window width w MUST be in the 2..10 range.
        """
        # The process consist in splitting the value in w-bit chunks,
        # then subtracting 2^w from chunks that are larger than 2^(w-1).
        # Each such subtraction yields a carry, added to the next chunk.
        # In order to ensure the requested properties, we do the following:
        #  - We add n to v initially, to get a value of 319 or 320 bits.
        #  - If the value is equal to +2^(w-1), we subtract 2^w and produce
        #    a carry (instead of keeping +2^(w-1) with no carry).
        #  - As an exception to the above, for the top digit, if we get
        #    2^(w-1) then we keep it.
        #
        # Since n is lower than but very close to 2^319, then when
        # v+n < 2^319 _and_ w divides 320, then the next-to-top digit
        # will produce a carry, and the value for the top digit will be
        # non-zero thanks to that carry. A contrario, for the maximum
        # value of v+n, 321 bits are always enough, even if w divides
        # 321 and there is a carry from below (thanks to the rule that
        # we keep +2^(w-1) if occurring on the top digit).
        #
        # All the computations in indexes below have static routing, so
        # they incur no cost in the VM model (for a given width w, they
        # could be completely unrolled).

        # Add n to the value. Final carry is ignored since we assume
        # that the source scalar was in the proper range.
        vv = []
        (x, c) = v[0].add32(self.n[0])
        vv.append(x)
        for i in range(1, 10):
            (x, c) = v[i].addc32(self.n[i], c)
            vv.append(x)

        # First digit (no incoming carry).
        tlen = (320 + w) // w
        tw = GFp(2**w)
        mw = GFp(2**w - 1)
        hw = GFp(2**(w - 1))
        t = []
        s = []
        d = vv[0].and32(mw)
        c = d.gte32(hw)
        d = c.select(d, tw - d)
        t.append(d)
        s.append(c)
        for i in range(1, tlen):
            # chunk i ranges from bit k1 in vv[j1] to bit k2 in vv[j2]
            k1 = i * w
            k2 = k1 + w - 1
            j1 = k1 >> 5
            k1 = k1 & 31
            j2 = k2 >> 5
            k2 = k2 & 31
            if j1 >= 10:
                d = c
            else:
                if j1 == j2:
                    if k1 == 0:
                        d = vv[j1].and32(mw)
                    elif k2 == 31:
                        d = vv[j1].shr32(k1)
                    else:
                        d = vv[j1].shr32(k1).and32(mw)
                elif j2 >= 10:
                    d = vv[j1].shr32(k1)
                else:
                    d = vv[j1].shr32(k1) + vv[j2].and32(GFp(2**(k2+1)-1)).shl32(w-1-k2)
                d = d + c
            if i < (tlen - 1):
                c = d.gte32(hw)
                d = c.select(d, tw - d)
            else:
                c = GFp.zero
            t.append(d)
            s.append(c)
        return (t, s)

    def lookup(self, win, idx, sgn):
        """
        Lookup a point in a window. win[i] contains point (i+1)*P;
        the provided index 'idx' (a GF(p) element) must contain a value
        between 0 and len(win) (inclusive). If idx is zero, this returns
        the point-at-infinity. Otherwise, it returns idx*P if sgn is zero,
        or -idx*P if sgn is one.
        """
        P = self.lookup_spec(win, idx)
        X = P.X
        Y = sgn.select(P.Y, -P.Y)
        return EcGFp5Impl.Point(self, X, Y, idx.iszero().bv_or(P.inf))

    def lookup_spec(self, win, idx):
        """
        Specialized variant of lookup(), when the index is known to be
        positive and non-zero.
        """
        X = win[0].X
        Y = win[0].Y
        inf = win[0].inf
        for i in range(1, len(win)):
            c = GFp(i + 1).equals(idx)
            X = c.select(X, win[i].X)
            Y = c.select(Y, win[i].Y)
        return EcGFp5Impl.Point(self, X, Y, inf)

    class Point:

        def __init__(self, curve, X, Y, inf):
            # We expect Weierstraß coordinates (X, Y).
            # Value 'inf' is a GF(p)-Boolean (1 for point-at-infinity).
            # This constructor assumes that the provided coordinates are
            # for a proper curve point in the right subgroup.
            self.curve = curve
            self.X = X
            self.Y = Y
            self.inf = inf

        def encode(self):
            w = self.Y / (self.curve.adiv3 - self.X)
            return self.inf.select(w, GFp5.zero)

        def __add__(self, P):
            if not(isinstance(P, EcGFp5Impl.Point)):
                return NotImplemented
            return self.add(P)

        def __radd__(self, P):
            if not(isinstance(P, EcGFp5Impl.Point)):
                return NotImplemented
            return P.add(self)

        def add(self, P):
            # Generic point addition routine: the code computes the
            # slope lambda for both the normal case and the special case
            # of a doubling, then selects the right one depending on
            # the input values. The point-at-infinity as input or output
            # is also handled with selection operations.
            X1 = self.X
            Y1 = self.Y
            inf1 = self.inf
            X2 = P.X
            Y2 = P.Y
            inf2 = P.inf
            sameX = X1.equals(X2)
            diffY = Y1.notequals(Y2)
            lambhi = sameX.select(Y2 - Y1,
                (3*X1.square()).addk01(self.curve.A0, self.curve.A1))
            lamblo = sameX.select(X2 - X1, 2*Y1)
            lamb = lambhi / lamblo
            X3 = lamb.square() - X1 - X2
            Y3 = lamb*(X1 - X3) - Y1
            inf3 = sameX.bv_and(diffY)
            X3 = inf1.select(X3, X2)
            X3 = inf2.select(X3, X1)
            Y3 = inf1.select(Y3, Y2)
            Y3 = inf2.select(Y3, Y1)
            inf3 = inf1.select(inf3, inf2)
            inf3 = inf2.select(inf3, inf1)
            return EcGFp5Impl.Point(self.curve, X3, Y3, inf3)

        def __sub__(self, P):
            if not(isinstance(P, EcGFp5Impl.Point)):
                return NotImplemented
            return self.add(-P)

        def __rsub__(self, P):
            if not(isinstance(P, EcGFp5Impl.Point)):
                return NotImplemented
            return P.add(-self)

        def __neg__(self):
            return EcGFp5Impl.Point(self.curve, self.X, -self.Y, self.inf)

        def double(self):
            # Since we work in a prime order subgroup, the double of
            # a non-infinity point cannot be the point-at-infinity. We
            # do computations under the assumption that the point is
            # not the point-at-infinity, and we reuse the 'inf' flag value.
            X = self.X
            Y = self.Y
            lamb = (3*X.square()).addk01(self.curve.A0, self.curve.A1) / (2*Y)
            X2 = lamb.square() - 2*X
            Y2 = lamb*(X - X2) - Y
            return EcGFp5Impl.Point(self.curve, X2, Y2, self.inf)

        def add_spec(self, P):
            """
            Specialized point addition routine, then it is guaranteed that
            no problematic case is encounted (i.e. P1 != 0, P2 != 0,
            P1 != P2 and P1 != -P2). If such a edge case is encounted,
            then an invalid point structure is returned. This function
            MUST NOT be used except in some specific contexts (see mul()).
            """
            X1 = self.X
            Y1 = self.Y
            X2 = P.X
            Y2 = P.Y
            lamb = (Y2 - Y1) / (X2 - X1)
            X3 = lamb.square() - X1 - X2
            Y3 = lamb*(X1 - X3) - Y1
            return EcGFp5Impl.Point(self.curve, X3, Y3, GFp.one)

        def __mul__(self, e):
            if isinstance(e, int):
                e = self.curve.int_to_limbs(e)
            elif not(isinstance(e, list)):
                return NotImplemented
            return self.mul(e)

        def __rmul__(self, e):
            if isinstance(e, int):
                e = self.curve.int_to_limbs(e)
            elif not(isinstance(e, list)):
                return NotImplemented
            return self.mul(e)

        def make_window(self, w):
            """
            Return points i*self for i in 1..2^(w-1). Returned array
            has length 2^(w-1); win[i] contains (i+1)*self.
            """
            # We can use add_spec() because if the point is not the
            # point-at-infinity, then all additions will be normal
            # cases. To cope with the case self == infinity, we simply
            # copy the current infinity flag.
            win = []
            win.append(self)
            win.append(self.double())
            for i in range(2, 2**(w-1)):
                P = win[i-1].add_spec(self)
                P = EcGFp5Impl.Point(self.curve, P.X, P.Y, self.inf)
                win.append(P)
            return win

        def mul(self, e):
            """
            Multiply this point by the scalar 'e'. The scalar is an
            array of ten GF(p) elements, containing 32-bit limbs in
            little-endian order; its value MUST be between 0 and n-1.
            """
            # We process the scalar by chunks of w bits (the "window").
            w = self.curve.window_width
            win = self.make_window(w)

            # Get the scalar in signed digits representation.
            (t, s) = self.curve.scalar_to_signed_padded(e, w)

            # Top scalar digit is non-zero and positive, we can
            # initialize our accumulator with a special lookup.
            P = self.curve.lookup_spec(win, t[len(t)-1])

            # Loop for all other digits. Since the accumulator (P) is
            # non-infinity, we statically know that for all iterations
            # except the two last ones, none of the additions may hit a
            # special case, unless the digit for that iteration is zero.
            # Hence, most point additions can use add_spec().
            for i in range(len(t) - 2, -1, -1):
                for j in range(0, w):
                    P = P.double()
                Q = self.curve.lookup(win, t[i], s[i])
                if i > 1:
                    c = t[i].iszero()
                    P2 = P.add_spec(Q)
                    P.X = c.select(P2.X, P.X)
                    P.Y = c.select(P2.Y, P.Y)
                else:
                    # Last iterations: we must first fix the 'inf' flag
                    # because we might have worked over the point-at-infinity
                    # all along, and we must set the current value to the
                    # point-at-infinity in that case.
                    # Furthermore, the last addition of a looked-up point
                    # may be a point doubling.
                    if i == 1:
                        P = EcGFp5Impl.Point(self.curve, P.X, P.Y, self.inf)
                    Q = self.curve.lookup(win, t[i], s[i])
                    P = P + Q

            return P

        def muladd(self, e, Q, f):
            """
            Given scalars e and f, compute e*self + f*Q. The scalars
            must be in the proper 0..n-1 range, and represented as
            ten 32-bit limbs each (see mul()).
            """
            # Initial lookups can use lookup_spec() because the top
            # digits of the two scalars are positive and non-zero.
            # All other operations use the generic variants.
            w = self.curve.window_width
            win1 = self.make_window(w)
            win2 = Q.make_window(w)
            (t1, s1) = self.curve.scalar_to_signed_padded(e, w)
            (t2, s2) = self.curve.scalar_to_signed_padded(f, w)
            P = self.curve.lookup_spec(win1, t1[len(t1)-1])
            P += self.curve.lookup_spec(win2, t2[len(t2)-1])
            for i in range(len(t1) - 2, -1, -1):
                for j in range(0, w):
                    P = P.double()
                P += self.curve.lookup(win1, t1[i], s1[i])
                P += self.curve.lookup(win2, t2[i], s2[i])
            return P

        def muladd_for_tests(self, e, Q, f):
            # Test-only function with the scalars provided as integers.
            return self.muladd(self.curve.int_to_limbs(e), Q, self.curve.int_to_limbs(f))

EcGFp5 = EcGFp5Impl()

# =========================================================================

import random

def TestGFp():
    # We only test Legendre and square roots; the rest of the code is
    # straightforward.
    p = 2**64 - 2**32 + 1
    z = GFp(7) # generator of GF(p)*, non-QR
    u = GFp(0)
    (r, c) = u.sqrt()
    if r != 0 or c != 1:
        raise Exception('failed on zero: r=%d c=%d' % (int(r), int(c)))
    for i in range(0, 100):
        while True:
            u = GFp(random.randint(0, p)).square() * z
            if u != 0:
                break
        (r, c) = u.sqrt()
        if r != 0 or c != 0:
            raise Exception('failed on non-QR: u=%d r=%d c=%d' % (int(u), int(r), int(c)))
        u = u.square()
        n1 = GFp.countOp
        (r, c) = u.sqrt()
        n2 = GFp.countOp
        if i == 0:
            print('sqrt-GF(p):', n2 - n1)
        if r.square() != u or c != 1:
            raise Exception('failed on QR: u=%d r=%d c=%d' % (int(u), int(r), int(c)))

def TestGFp5():
    # Some test vectors obtained from Sage.
    a = GFp5((9788683869780751860, 18176307314149915536, 17581807048943060475, 16706651231658143014, 424516324638612383))
    b = GFp5((1541862605911742196, 5168181287870979863, 10854086836664484156, 11043707160649157424, 943499178011708365))
    apb = GFp5((11330546475692494056, 4897744532606311078, 9989149816192960310, 9303614322892716117, 1368015502650320748))
    amb = GFp5((8246821263869009664, 13008126026278935673, 6727720212278576319, 5662944071008985590, 17927761216041488339))
    atb = GFp5((5924286846078684570, 12564682493825924142, 17116577152380521223, 5260948460973948760, 15673927150284637712))
    adb = GFp5((6854214528917216670, 4676163378868226016, 7338977912708396522, 9922012834063967541, 11413717840889184601))

    n1 = GFp.countOp
    c = a + b
    n2 = GFp.countOp
    print("add:", n2 - n1)
    if apb != c: 
        raise Exception('failed add')

    n1 = GFp.countOp
    c = a - b
    n2 = GFp.countOp
    print("sub:", n2 - n1)
    if amb != c:
        raise Exception('failed sub')

    n1 = GFp.countOp
    c = a * b
    n2 = GFp.countOp
    print("mul:", n2 - n1)
    if atb != c:
        raise Exception('failed mul')

    n1 = GFp.countOp
    c = a.square()
    n2 = GFp.countOp
    print("sqr:", n2 - n1)
    if (a * a) != c:
        raise Exception('failed sqr')

    n1 = GFp.countOp
    c = b.invert()
    n2 = GFp.countOp
    print("inv:", n2 - n1)
    if adb != (a * c):
        raise Exception('failed inv')

    n1 = GFp.countOp
    c = a / b
    n2 = GFp.countOp
    print("div:", n2 - n1)
    if adb != (a / b):
        raise Exception('failed div')

    # Generate some random squares and non-squares to test is_square().
    if GFp5((0, 0, 0, 0, 0)).legendre() != 0:
        raise Exception('QR: zero')
    (s, c) = GFp5((0, 0, 0, 0, 0)).sqrt()
    if s != 0 or c != 1:
        raise Exception('sqrt: zero')
    p = 2**64 - 2**32 + 1
    for i in range(0, 100):
        a = GFp5((random.randint(0, p), random.randint(0, p), random.randint(0, p), random.randint(0, p), random.randint(0, p)))
        a = a.square()
        b = a * 7
        n1 = GFp.countOp
        aleg = a.legendre()
        n2 = GFp.countOp
        if i == 0:
            print("legendre:", n2 - n1)
        n1 = GFp.countOp
        (asqrt, acc) = a.sqrt()
        n2 = GFp.countOp
        if i == 0:
            print("sqrt:", n2 - n1)
        bleg = b.legendre()
        (bsqrt, bcc) = b.sqrt()
        if a == GFp5.zero:
            if aleg != 0 or bleg != 0:
                raise Exception('wrong Legendre symbol of zero')
            if asqrt != 0 or bsqrt != 0:
                raise Exception('wrong square root of zero')
            if acc != 1 or bcc != 1:
                raise Exception('square root of zero failed')
        else:
            if aleg != 1:
                raise Exception('QR declared non-square')
            if bleg != GFp.minusone:
                raise Exception('non-QR declared square')
            if acc != 1:
                raise Exception('square root of QR failed')
            if asqrt.square() != a:
                raise Exception('wrong square root value returned')
            if bsqrt != 0 or bcc != 0:
                raise Exception('square root of non-QR should have failed')

def TestEcGFp5():
    # Test vectors generated with Sage.
    # P0 is group neutral.
    # P1 is a random point in the group (encoded as w1)
    # P2 = e*P1 (encoded as w2)
    # P3 = P1 + P2 (encoded as w3)
    # P4 = 2*P1 (encoded as w4)
    # P5 = 2*P2 (encoded as w5)
    # P6 = 2*P1 + P2 (encoded as w6)
    # P7 = P1 + 2*P2 (encoded as w7)
    # All operations here are in the group of non-n-torsion points of the
    # curve.

    w0 = GFp5((0, 0, 0, 0, 0))
    w1 = GFp5((12539254003028696409, 15524144070600887654, 15092036948424041984, 11398871370327264211, 10958391180505708567))
    w2 = GFp5((11001943240060308920, 17075173755187928434, 3940989555384655766, 15017795574860011099, 5548543797011402287))
    w3 = GFp5((246872606398642312, 4900963247917836450, 7327006728177203977, 13945036888436667069, 3062018119121328861))
    w4 = GFp5((8058035104653144162, 16041715455419993830, 7448530016070824199, 11253639182222911208, 6228757819849640866))
    w5 = GFp5((10523134687509281194, 11148711503117769087, 9056499921957594891, 13016664454465495026, 16494247923890248266))
    w6 = GFp5((12173306542237620, 6587231965341539782, 17027985748515888117, 17194831817613584995, 10056734072351459010))
    w7 = GFp5((9420857400785992333, 4695934009314206363, 14471922162341187302, 13395190104221781928, 16359223219913018041))

    # These values should not decode successfully.
    bww = [ GFp5((13557832913345268708, 15669280705791538619, 8534654657267986396, 12533218303838131749, 5058070698878426028)),
            GFp5((135036726621282077, 17283229938160287622, 13113167081889323961, 1653240450380825271, 520025869628727862)),
            GFp5((6727960962624180771, 17240764188796091916, 3954717247028503753, 1002781561619501488, 4295357288570643789)),
            GFp5((4578929270179684956, 3866930513245945042, 7662265318638150701, 9503686272550423634, 12241691520798116285)),
            GFp5((16890297404904119082, 6169724643582733633, 9725973298012340311, 5977049210035183790, 11379332130141664883)),
            GFp5((13777379982711219130, 14715168412651470168, 17942199593791635585, 6188824164976547520, 15461469634034461986)) ]

    e = 841809598287430541331763924924406256080383779033370172527955679319982746101779529382447999363236

    if not(EcGFp5.validate(w0)):
        raise Exception('cannot validate w0')
    if not(EcGFp5.validate(w1)):
        raise Exception('cannot validate w1')
    if not(EcGFp5.validate(w2)):
        raise Exception('cannot validate w2')
    if not(EcGFp5.validate(w3)):
        raise Exception('cannot validate w3')
    if not(EcGFp5.validate(w4)):
        raise Exception('cannot validate w4')
    if not(EcGFp5.validate(w5)):
        raise Exception('cannot validate w5')
    if not(EcGFp5.validate(w6)):
        raise Exception('cannot validate w6')
    if not(EcGFp5.validate(w7)):
        raise Exception('cannot validate w7')

    (P0, c) = EcGFp5.decode(w0)
    if not(c.asbool()):
        raise Exception('could not decode point P0')
    if P0.encode() != w0:
        raise Exception('decode/encode failed for P0')
    if not(P0.inf.asbool()):
        raise Exception('wrong internal representation of point-at-infinity')
    (P1, c) = EcGFp5.decode(w1)
    if not(c.asbool()):
        raise Exception('could not decode point P1')
    if P1.encode() != w1:
        raise Exception('decode/encode failed for P1')
    (P2, c) = EcGFp5.decode(w2)
    if not(c.asbool()):
        raise Exception('could not decode point P2')
    if P2.encode() != w2:
        raise Exception('decode/encode failed for P2')
    (P3, c) = EcGFp5.decode(w3)
    if not(c.asbool()):
        raise Exception('could not decode point P3')
    if P3.encode() != w3:
        raise Exception('decode/encode failed for P3')
    (P4, c) = EcGFp5.decode(w4)
    if not(c.asbool()):
        raise Exception('could not decode point P4')
    if P4.encode() != w4:
        raise Exception('decode/encode failed for P4')
    (P5, c) = EcGFp5.decode(w5)
    if not(c.asbool()):
        raise Exception('could not decode point P5')
    if P5.encode() != w5:
        raise Exception('decode/encode failed for P5')
    (P6, c) = EcGFp5.decode(w6)
    if not(c.asbool()):
        raise Exception('could not decode point P6')
    if P6.encode() != w6:
        raise Exception('decode/encode failed for P6')
    (P7, c) = EcGFp5.decode(w7)
    if not(c.asbool()):
        raise Exception('could not decode point P7')
    if P7.encode() != w7:
        raise Exception('decode/encode failed for P7')

    for bw in bww:
        if EcGFp5.validate(bw) != 0:
            raise Exception('invalid value declared decodable')
        (P, c) = EcGFp5.decode(bw)
        if c != 0 or P.inf != 1:
            raise Exception('invalid value was decoded')

    Q3 = P1 + P2
    if Q3.encode() != w3:
        raise Exception('failed KAT (P3)')
    Q4 = P1.double()
    if Q4.encode() != w4:
        raise Exception('failed KAT (P4)')
    Q4 = P1 + P1
    if Q4.encode() != w4:
        raise Exception('failed KAT (P4) (add)')
    Q5 = P2.double()
    if Q5.encode() != w5:
        raise Exception('failed KAT (P5)')
    Q5 = P2 + P2
    if Q5.encode() != w5:
        raise Exception('failed KAT (P5) (add)')
    Q6 = P4 + P2
    if Q6.encode() != w6:
        raise Exception('failed KAT (P6)')
    Q7 = P5 + P1
    if Q7.encode() != w7:
        raise Exception('failed KAT (P7)')

    Q = P1 + P0
    if Q.encode() != w1:
        raise Exception('failed add zero (1)')
    Q = P0 + P1
    if Q.encode() != w1:
        raise Exception('failed add zero (2)')
    Q = P0 + P0
    if Q.encode() != w0:
        raise Exception('failed add zero (3)')
    Q = P1 - P1
    if Q.encode() != w0:
        raise Exception('failed subtract self')
    Q = P1 + (-P1)
    if Q.encode() != w0:
        raise Exception('failed add opposite')
    Q = P1 - (-P2)
    if Q.encode() != w3:
        raise Exception('failed subtract')

    cc1 = GFp.countOp
    EcGFp5.validate(w1)
    cc2 = GFp.countOp
    print('validate:', cc2 - cc1)
    cc1 = GFp.countOp
    EcGFp5.decode(w1)
    cc2 = GFp.countOp
    print('decode:', cc2 - cc1)
    cc1 = GFp.countOp
    P1.encode()
    cc2 = GFp.countOp
    print('encode:', cc2 - cc1)
    cc1 = GFp.countOp
    P1 + P2
    cc2 = GFp.countOp
    print('add:', cc2 - cc1)
    cc1 = GFp.countOp
    P1.add_spec(P2)
    cc2 = GFp.countOp
    print('add_spec:', cc2 - cc1)
    cc1 = GFp.countOp
    P1.double()
    cc2 = GFp.countOp
    print('double:', cc2 - cc1)

    cc1 = GFp.countOp
    Q2 = e*P1
    cc2 = GFp.countOp
    print('mul:', cc2 - cc1)
    if Q2.encode() != w2:
        raise Exception('failed KAT (mul)')

    n = 1067993516717146951041484916571792702745057740581727230159139685185762082554198619328292418486241
    for i in range(0, 10):
        if i < 4:
            if (i & 1) == 0:
                e1 = 0
            else:
                e1 = random.randint(0, n)
            if i < 2:
                e2 = 0
            else:
                e2 = random.randint(0, n)
        else:
            e1 = random.randint(0, n)
            e2 = random.randint(0, n)
        R1 = e1*P1 + e2*P2
        cc1 = GFp.countOp
        R2 = P1.muladd_for_tests(e1, P2, e2)
        cc2 = GFp.countOp
        if i == 0:
            print('muladd:', cc2 - cc1)
        if R1.encode() != R2.encode():
            raise Exception('failed muladd')
        print('.', flush=True, end='')
    print()
