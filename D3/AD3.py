import sympy as sp
t = sp.symbols('t', real=True)

B = [
    (-131.66*t**3+108.72*t**2+154.59*t-135.39, -239.05*t**3+490.74*t**2-405.96*t+171.46),
    (-123.0*t**3+222.54*t**2-28.98*t-3.74, -197.3*t**3+353.04*t**2-179.01*t+17.19),
    (-12.6*t**2+24.4*t+66.82, -9.74*t**2+1.58*t-6.08),
    (0.22*t**2-12.32*t+78.62, -4.15*t**2+4.06*t-14.24),
    (-137.21*t**3+182.25*t**2-109.11*t+66.52, -4.68*t**3+4.86*t**2-38.91*t-14.33),
    (14.09*t**2+10.24*t+2.45, 0.5*t**2-21.82*t-53.06),
    (27.44*t**2-153.52*t+26.78, 48.55*t**2-38.0*t-74.38),
    (872.4*t**3-1497.24*t**2+588.75*t-99.3, 353.52*t**3-304.98*t**2+186.75*t-63.83)
]

area_integrals = []
for i,(xexpr, yexpr) in enumerate(B, start=1):
    dydt = sp.diff(yexpr, t)
    integrand = sp.simplify(xexpr * dydt)
    integral = sp.integrate(integrand, (t,0,1))
    area_integrals.append(integral)
    print("Segment {}: integral = {:.12f}".format(i, float(sp.N(integral, 20))))

total = sum(area_integrals)
print("Area=", float(sp.N(total,20)))

