import math
import time
import numpy as np

def ik_3R(L1, L2, L3, target:np.array, tol=1e-9) -> list:
    X = target[0]
    Y = target[1]
    Z = target[2]
    # Step 1: compute D
    S = X**2 + Y**2 + Z**2
    D = S - (L1**2 + L2**2 + L3**2)
    
    # Step 2: quadratic coefficients for u = cos(q3)
    a = 4 * L3**2 * (L2**2 - L1**2)
    b = -4 * L2 * L3 * (D + 2 * L1**2)
    c = D**2 - 4 * L1**2 * L2**2 + 4 * L1**2 * X**2

    solutions = []

    # Step 3: solve quadratic
    if abs(a) < tol:  # degenerate to linear
        if abs(b) < tol:
            return []  # no equation to solve
        u_roots = [-c / b]
    else:
        disc = b**2 - 4*a*c
        if disc < -tol:
            return []  # no real solutions
        elif abs(disc) < tol:
            u_roots = [-b / (2*a)]
        else:
            sqrt_disc = math.sqrt(max(disc, 0.0))
            u_roots = [(-b + sqrt_disc) / (2*a),
                       (-b - sqrt_disc) / (2*a)]

    # Step 4: loop over cos(q3) solutions
    for u in u_roots:
        if abs(u) > 1 + 1e-9:
            continue  # invalid
        u = max(-1, min(1, u))  # clamp

        for s3_sign in [1, -1]:
            s3 = s3_sign * math.sqrt(max(0.0, 1 - u**2))
            C = L2 + L3 * u

            if abs(C) < tol:
                continue  # singular, skip or handle separately

            # q2 from atan2 form
            num = -X
            den = (D - 2 * L2 * L3 * u) / (2 * L1)
            q2 = math.atan2(num, den)

            # compute A, B
            A = L1 + C * math.cos(q2)
            B = L3 * s3

            # q1
            q1 = math.atan2(Y, -Z) - math.atan2(B, A)

            # q3
            q3 = math.atan2(s3, u)

            solutions.append((q1, q2, q3))

    return solutions

def fk_3R(L1, L2, L3, q1, q2, q3) -> np.array:
    x = -L2*math.sin(q2) - L3*math.sin(q2)*math.cos(q3)
    y = L1*math.sin(q1) + L2*math.sin(q1)*math.cos(q2) + L3*math.sin(q1)*math.cos(q2)*math.cos(q3) + L3*math.sin(q3)*math.cos(q1)
    z = -L1*math.cos(q1) - L2*math.cos(q1)*math.cos(q2) + L3*math.sin(q1)*math.sin(q3) - L3*math.cos(q1)*math.cos(q2)*math.cos(q3)
    return np.array([x, y, z])


if __name__ == "__main__":
    L_1 = 0.110
    L_2 = 0.317
    L_3 = 0.330
    # Target position
    target = np.array([0.1, 0.3, -0.5])

    start_time = time.time()
    sols = ik_3R(L_1, L_2, L_3, target)
    elapsed_time = (time.time() - start_time) * 1000
    print(f'IK time: {elapsed_time} ms \t {elapsed_time*1000} us')
    if not sols:
        print('No solutions found')
    for i, (q1, q2, q3) in enumerate(sols):
        print(f"Solution {i+1}: q1={q1:.4f}, q2={q2:.4f}, q3={q3:.4f}")
        pos = fk_3R(L_1, L_2, L_3, q1, q2, q3)
        dif = np.linalg.norm(pos-target)
        print(f'Difference: {dif}')