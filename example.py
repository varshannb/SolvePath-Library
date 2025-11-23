from solvepath import comb, gcd

res, steps = comb(10, 3, show_steps=True)
print("comb(10,3) =", res)
print("Steps:")
for s in steps:
    print(" -", s)

g, gsteps = gcd(48, 18, show_steps=True)
print("\ngcd(48,18) =", g)
for s in gsteps:
    print(" *", s)
