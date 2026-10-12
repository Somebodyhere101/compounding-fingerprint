# Worked forcing-class example (Section 3.2)

Order 5, labels e a b c d; cue: the row and column of e, a*a = b, a*b = c; query b*b.

- Latin squares of order 5: 161,280
- consistent with the cue: 6 Latin squares, 6 loops, 1 group table(s)
- values of b*b: Latin ['a', 'd', 'e'], loop ['a', 'd', 'e'], group ['d'] -> G-forced
- a*c over all consistent Latin squares: ['d'] -> L-forced
- printed loop: b*b = e, (a*a)*b = e, a*(a*b) = d; associative: False
- commutative loops among those consistent with the cue: 1
- engine: latin 6 tables, b*b in ['e', 'a', 'd']; loop 6 tables, b*b in ['e', 'a', 'd']; iso:Z5 1 tables, b*b in ['d']
