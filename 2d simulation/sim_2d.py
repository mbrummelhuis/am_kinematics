import sympy as sp
import numpy as np
from sympy import symbols, Function, cos, sin, diff


# Time variable
t = symbols('t')

# Other variables
mb, mm, Ib, Im, Lb, Lm, g = symbols('m_b m_m I_b I_m L_b L_m g')

# Generalized coordinates as functions of time
xb = Function('x_b')(t)
yb = Function('y_b')(t)
thetab = Function('theta_b')(t)
q = Function('q')(t)

# Derivatives
xb_dot = diff(xb, t)
yb_dot = diff(yb, t)
thetab_dot = diff(thetab, t)
q_dot = diff(q, t)

xb_ddot = diff(xb_dot, t)
yb_ddot = diff(yb_dot, t)
thetab_ddot = diff(thetab_dot, t)
q_ddot = diff(q_dot, t)

# End-effector coordinates
xe = Function('x_e')(t)
ye = Function('y_e')(t)
thetae = Function('theta_e')(t)

xe_dot = diff(xe, t)
ye_dot = diff(ye, t)
thetae_dot = diff(thetae, t)

# Deriving the kinematic equation
xe = xb + Lm*cos(thetab+q)
ye = yb + Lm*sin(thetab+q)
thetae = thetab + q
k = sp.Matrix([xe, ye, thetae]) # kinematic equation
sp.pprint(k)

# Jacobian
J_A = k.jacobian([xb, yb, thetab, q])
J_A = J_A.subs({Lm:1})
print(J_A.shape)

# Splitting into controlled and uncontrolled parts
J_A_controlled = J_A[:,0:2].row_join(J_A[:,3])
J_A_uncontrolled = J_A[:,2]

# Dynamics
# Deriving the kinetic energy
T_b = mb*(xb_dot**2 + yb_dot**2)/2 + Ib*thetab_dot**2/2
T_m = mm*(xb_dot**2 + yb_dot**2 + (Lm*q_dot**2)/4 + (Lm*q_dot*thetab_dot)/2 + (Lm*thetab_dot**2)/4.)/2 + Im*(q_dot+thetab_dot)**2/2

# Deriving the potential energy
U_b = mb*g*yb
U_m = mm*g*yb + mm*g*Lm*sin(q+thetab)/2

# Deriving the Lagrangian
L = T_b + T_m - U_b - U_m

# Deriving the equations of motion
eq1 = diff(diff(L, xb_dot), t) - diff(L, xb)
eq2 = diff(diff(L, yb_dot), t) - diff(L, yb)
eq3 = diff(diff(L, thetab_dot), t) - diff(L, thetab)
eq4 = diff(diff(L, q_dot), t) - diff(L, q)

sp.pprint("Equations of motion")
sp.pprint(eq1)
sp.pprint(eq2)
sp.pprint(eq3)
sp.pprint(eq4)

# Allocation matrix
N = sp.Matrix([cos(thetab), cos(thetab), 0],
          [sin(thetab), sin(thetab), 0],
          [-Lb/2, Lb/2, 0],
          [0, 0, 1]).subs({Lb:1.})