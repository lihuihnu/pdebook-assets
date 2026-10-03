"""Steady 2D convection-diffusion: transparent standard-library baseline.

The smooth manufactured solution is used only for prescribed data and error
measurement. All interior values are obtained by solving the discrete system.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import platform


def coefficients(h, kappa, bx, by, scheme="upwind"):
    if not all(math.isfinite(v) for v in (h, kappa, bx, by)):
        raise ValueError("non-finite coefficient")
    if h <= 0 or kappa <= 0 or bx < 0 or by < 0:
        raise ValueError("positive h and kappa, nonnegative velocities required")
    d = kappa/h**2
    if scheme == "upwind":
        west, east = d + bx/h, d
        south, north = d + by/h, d
    elif scheme == "central":
        west, east = d + bx/(2*h), d - bx/(2*h)
        south, north = d + by/(2*h), d - by/(2*h)
    else:
        raise ValueError("unknown convection scheme")
    return west + east + south + north, west, east, south, north


def layer_profile(x, kappa):
    small = math.exp(-1/kappa)
    return (math.exp((x-1)/kappa)-small)/(1-small)


def reference(problem, x, y, kappa):
    if problem == "smooth":
        return x + y + math.sin(math.pi*x)*math.sin(math.pi*y)
    if problem == "bilinear":
        return x + y + x*y
    if problem == "layer":
        return layer_profile(x, kappa)
    raise ValueError("unknown problem")


def boundary(problem, x, y, kappa):
    if problem == "smooth":
        return x + y
    return reference(problem, x, y, kappa)


def source(problem, x, y, kappa, bx, by):
    if problem == "smooth":
        sx, sy = math.sin(math.pi*x), math.sin(math.pi*y)
        cx, cy = math.cos(math.pi*x), math.cos(math.pi*y)
        return (2*kappa*math.pi**2*sx*sy + bx*(1+math.pi*cx*sy)
                + by*(1+math.pi*sx*cy))
    if problem == "bilinear":
        return bx*(1+y) + by*(1+x)
    if problem == "layer":
        return 0.0
    raise ValueError("unknown problem")


def setup(problem, n, kappa, bx, by):
    if type(n) is not int or n < 2:
        raise ValueError("N must be an integer at least 2")
    coefficients(1/n, kappa, bx, by)
    h = 1/n
    u = [[0.0]*(n+1) for j in range(n+1)]
    f = [[0.0]*(n+1) for j in range(n+1)]
    for j in range(n+1):
        for i in range(n+1):
            if i == 0 or i == n or j == 0 or j == n:
                u[j][i] = boundary(problem, i*h, j*h, kappa)
            else:
                f[j][i] = source(problem, i*h, j*h, kappa, bx, by)
    return u, f


def upwind_sweep(u, f, h, kappa, bx, by):
    n = len(u)-1
    d = kappa/h**2
    west, east = d + bx/h, d
    south, north = d + by/h, d
    diagonal = west + east + south + north
    change = 0.0
    for j in range(1, n):
        for i in range(1, n):
            old = u[j][i]
            u[j][i] = (f[j][i] + west*u[j][i-1] + east*u[j][i+1]
                       + south*u[j-1][i] + north*u[j+1][i])/diagonal
            change = max(change, abs(u[j][i]-old))
    return change


def residual(u, f, weights):
    diagonal, west, east, south, north = weights
    n = len(u)-1
    values = []
    for j in range(1, n):
        for i in range(1, n):
            applied = (diagonal*u[j][i] - west*u[j][i-1] - east*u[j][i+1]
                       - south*u[j-1][i] - north*u[j+1][i])
            values.append(f[j][i]-applied)
    return values


def assemble(u, f, weights):
    """Small-system diagnostic only; positive weights need not be assumed."""
    n = len(u)-1
    width = n-1
    size = width**2
    matrix = [[0.0]*size for row in range(size)]
    rhs = [0.0]*size
    diagonal, west, east, south, north = weights
    for j in range(1, n):
        for i in range(1, n):
            row = (j-1)*width + i-1
            matrix[row][row] = diagonal
            rhs[row] = f[j][i]
            for ni, nj, weight in [(i-1,j,west), (i+1,j,east),
                                   (i,j-1,south), (i,j+1,north)]:
                if ni == 0 or ni == n or nj == 0 or nj == n:
                    rhs[row] += weight*u[nj][ni]
                else:
                    col = (nj-1)*width + ni-1
                    matrix[row][col] -= weight
    return matrix, rhs


def pivoted_solve(matrix, rhs):
    """Ordinary partial-pivot Gaussian elimination for the 81-node probe."""
    a = [row.copy() + [value] for row, value in zip(matrix, rhs)]
    size = len(rhs)
    for k in range(size):
        pivot = max(range(k, size), key=lambda i: abs(a[i][k]))
        if a[pivot][k] == 0:
            raise ArithmeticError("singular diagnostic system")
        a[k], a[pivot] = a[pivot], a[k]
        for i in range(k+1, size):
            factor = a[i][k]/a[k][k]
            a[i][k] = 0.0
            for j in range(k+1, size+1):
                a[i][j] -= factor*a[k][j]
    answer = [0.0]*size
    for i in range(size-1, -1, -1):
        value = a[i][size]
        for j in range(i+1, size):
            value -= a[i][j]*answer[j]
        answer[i] = value/a[i][i]
    if not all(math.isfinite(x) for x in answer):
        raise ArithmeticError("non-finite diagnostic solution")
    return answer


def solve(problem, n, kappa, bx, by, scheme, tolerance, max_sweeps):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite residual tolerance required")
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("positive integer sweep budget required")
    u, f = setup(problem, n, kappa, bx, by)
    weights = coefficients(1/n, kappa, bx, by, scheme)
    history, states = [], []
    initial = max(abs(r) for r in residual(u, f, weights))
    history.append((0, initial, 0.0))
    if scheme == "central":
        if n > 16:
            raise ValueError("dense diagnostic restricted to N <= 16")
        matrix, rhs = assemble(u, f, weights)
        answer = pivoted_solve(matrix, rhs)
        for j in range(1, n):
            for i in range(1, n):
                u[j][i] = answer[(j-1)*(n-1)+i-1]
        norm = max(abs(r) for r in residual(u, f, weights))
        history.append((1, norm, max(abs(x) for x in answer)))
        if norm > tolerance:
            raise ArithmeticError("diagnostic solve failed residual check")
    else:
        for sweep in range(1, max_sweeps+1):
            change = upwind_sweep(u, f, 1/n, kappa, bx, by)
            norm = max(abs(r) for r in residual(u, f, weights))
            if not math.isfinite(norm) or not math.isfinite(change):
                raise ArithmeticError("non-finite Gauss-Seidel iterate")
            history.append((sweep, norm, change))
            if problem == "bilinear" and n == 4 and sweep <= 3:
                for j in range(1, n):
                    for i in range(1, n):
                        states.append((sweep,i,j,u[j][i]))
            if norm <= tolerance:
                break
        else:
            raise ArithmeticError("Gauss-Seidel exhausted its sweep budget")
    return u, f, weights, history, states


def numerical_flux(left, right, velocity, kappa, h, scheme):
    face_value = left if scheme == "upwind" else (left+right)/2
    return velocity*face_value - kappa*(right-left)/h


def formatted(value):
    return format(value, ".17g") if isinstance(value, float) else value


def write_csv(path, headers, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([formatted(value) for value in row])


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    bx, by, kappa = config["bx"], config["by"], config["kappa"]
    tolerance, budget = config["residual_tolerance"], config["max_sweeps"]
    jobs = []
    for problem, grids in [("smooth",config["smooth_grids"]),
                           ("bilinear",config["bilinear_grids"])]:
        for n in grids:
            jobs.append((f"{problem}-{n}",problem,n,kappa,"upwind",tolerance))
    for tol in config["tolerance_study"]["tolerances"]:
        n = config["tolerance_study"]["N"]
        jobs.append((f"tolerance-{tol:g}","smooth",n,kappa,"upwind",tol))
    for scheme in config["layer"]["schemes"]:
        jobs.append((f"layer-{scheme}","layer",config["layer"]["N"],
                     config["layer"]["kappa"],scheme,tolerance))
    summary, nodes, histories, faces, balances, operators, states = [],[],[],[],[],[],[]
    small_matrix, small_rhs, solutions = [], [], {}
    for identity, problem, n, diffusivity, scheme, tol in jobs:
        u, f, weights, history, samples = solve(problem,n,diffusivity,bx,by,scheme,tol,budget)
        solutions[identity] = u
        h = 1/n
        errors, node_rows = [], []
        for j in range(n+1):
            for i in range(n+1):
                exact = reference(problem,i*h,j*h,diffusivity)
                edge = int(i in (0,n) or j in (0,n))
                error = u[j][i]-exact
                node_rows.append((identity,i,j,i*h,j*h,u[j][i],exact,error,edge,f[j][i]))
                if not edge:
                    errors.append(error)
        nodes.extend(node_rows)
        operators.append((identity,n,scheme,diffusivity,bx,by,*weights))
        for row in history:
            histories.append((identity,*row))
        for row in samples:
            states.append((identity,*row))
        fx, fy = {}, {}
        for j in range(1,n):
            for i in range(n):
                value = numerical_flux(u[j][i],u[j][i+1],bx,diffusivity,h,scheme)
                fx[i,j] = value
                faces.append((identity,"x",i,j,(i+.5)*h,j*h,value))
        for j in range(n):
            for i in range(1,n):
                value = numerical_flux(u[j][i],u[j+1][i],by,diffusivity,h,scheme)
                fy[i,j] = value
                faces.append((identity,"y",i,j,i*h,(j+.5)*h,value))
        defects, volume_source = [], []
        for j in range(1,n):
            for i in range(1,n):
                net = h*(fx[i,j]-fx[i-1,j]+fy[i,j]-fy[i,j-1])
                supplied = h*h*f[j][i]
                defect = net-supplied
                defects.append(defect);volume_source.append(supplied)
                balances.append((identity,i,j,net,supplied,defect))
        boundary_net = h*(math.fsum(fx[n-1,j]-fx[0,j] for j in range(1,n))
                          +math.fsum(fy[i,n-1]-fy[i,0] for i in range(1,n)))
        source_sum = math.fsum(volume_source)
        algebraic_bound = (2/(bx+by)*history[-1][1]
                           if scheme == "upwind" and bx+by > 0 else "")
        flat = [value for row in u for value in row]
        summary.append([identity,problem,n,scheme,diffusivity,tol,history[-1][0],
                        history[-1][1],algebraic_bound,max(abs(e) for e in errors),
                        h*math.sqrt(math.fsum(e*e for e in errors)),min(flat),max(flat),
                        bx*h/diffusivity,by*h/diffusivity,max(abs(d) for d in defects),
                        boundary_net,source_sum,boundary_net-source_sum])
        if identity == "bilinear-4":
            matrix,rhs = assemble(u,f,weights)
            for row in range(len(rhs)):
                small_rhs.append((row,rhs[row]))
                for col,value in enumerate(matrix[row]):
                    if value:
                        small_matrix.append((row,col,value))
    refinement = []
    for n in config["smooth_grids"]:
        parent = next((row for row in summary if row[0] == f"smooth-{n//2}"),None)
        current = next(row for row in summary if row[0] == f"smooth-{n}")
        if parent:
            order = math.log(parent[9]/current[9],2)
            coarse,fine = solutions[parent[0]],solutions[current[0]]
            difference = max(abs(coarse[j][i]-fine[2*j][2*i])
                             for j in range(1,n//2) for i in range(1,n//2))
            refinement.append((n//2,n,parent[9],current[9],order,difference))
    tolerance_rows = []
    reference_grid = solutions[f"smooth-{config['tolerance_study']['N']}"]
    for row in summary:
        if row[0].startswith("tolerance-"):
            grid = solutions[row[0]]
            difference = max(abs(grid[j][i]-reference_grid[j][i])
                             for j in range(1,len(grid)-1) for i in range(1,len(grid)-1))
            tolerance_rows.append((row[0],row[5],row[6],row[7],row[8],difference,row[9]))
    fields = "case,problem,N,scheme,kappa,tolerance,sweeps,residual_inf,algebraic_bound,error_inf,error_l2,min_value,max_value,Px,Py,local_defect_inf,boundary_net,source_sum,global_defect".split(",")
    write_csv(output/"summary.csv",fields,summary)
    write_csv(output/"nodes.csv","case,i,j,x,y,value,exact,error,boundary,source".split(","),nodes)
    write_csv(output/"history.csv","case,sweep,residual_inf,change_inf".split(","),histories)
    write_csv(output/"faces.csv","case,axis,i,j,x,y,flux".split(","),faces)
    write_csv(output/"balances.csv","case,i,j,net_flux,volume_source,defect".split(","),balances)
    write_csv(output/"operator.csv","case,N,scheme,kappa,bx,by,diagonal,west,east,south,north".split(","),operators)
    write_csv(output/"iteration_states.csv","case,sweep,i,j,value".split(","),states)
    write_csv(output/"small_matrix.csv",["row","col","value"],small_matrix)
    write_csv(output/"small_rhs.csv",["row","value"],small_rhs)
    write_csv(output/"refinement.csv","coarse_N,fine_N,coarse_error,fine_error,observed_order,common_node_difference".split(","),refinement)
    write_csv(output/"tolerance.csv","case,tolerance,sweeps,residual_inf,algebraic_bound,tight_grid_difference,error_inf".split(","),tolerance_rows)
    layer_reference = [(i/1000,layer_profile(i/1000,config["layer"]["kappa"]))
                       for i in range(1001)]
    write_csv(output/"layer_reference.csv",["x","exact"],layer_reference)
    log = [f"Python {platform.python_version()}; standard-library numerical core.",
           "All interior values obtained by Gauss-Seidel or the small central diagnostic solve.",
           "Residual uses the unscaled PDE stencil; tolerance is absolute.",
           "Flux balance covers [h/2,1-h/2]^2 and point-sampled source volumes."]
    for row in summary:
        log.append(f"{row[0]}: solves=1, sweeps={row[6]}, residual={row[7]:.8g}, error={row[9]:.8g}")
    (output/"run.txt").write_text("\n".join(log)+"\n")
    print("\n".join(log))


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here/"config/problems.json")
    parser.add_argument("--output", type=Path, default=here/"results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)
