#!/usr/bin/env python3
"""Small, explicit P1 Poisson examples on the unit square; standard library only."""
import argparse
import csv
import itertools
import json
import math
import platform
from pathlib import Path

CASES = ("constant_zero", "constant_lift", "affine_patch", "linear_source", "quartic", "mixed_affine")


def solve_four_triangles(boundary):
    right = 1.0 / 3.0
    for value in boundary:
        right -= (-1.0) * value
    return right / 4.0


def triangle_geometry(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    d = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    if d == 0.0:
        raise ValueError("degenerate triangle")
    gradients = [( (y2-y3)/d, (x3-x2)/d ),
                 ( (y3-y1)/d, (x1-x3)/d ),
                 ( (y1-y2)/d, (x2-x1)/d )]
    return abs(d)/2.0, gradients


def shape_values(points, gradients, x, y):
    values = []
    for i in range(3):
        gx, gy = gradients[i]
        px, py = points[i]
        values.append(1.0 + gx*(x-px) + gy*(y-py))
    return values


def local_matrix(area, gradients):
    matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            dot = gradients[i][0]*gradients[j][0] + gradients[i][1]*gradients[j][1]
            row.append(area*dot)
        matrix.append(row)
    return matrix


def source(case, x, y):
    if case in ("constant_zero", "constant_lift"):
        return 1.0
    if case == "linear_source":
        return x+y
    if case == "quartic":
        return 2.0*(x*(1.0-x)+y*(1.0-y))
    return 0.0


def exact(case, x, y):
    if case in ("affine_patch", "mixed_affine"):
        return 1.0+x+2.0*y
    if case == "quartic":
        return x*(1.0-x)*y*(1.0-y)
    return None


def boundary_value(case, x, y):
    if case in ("constant_lift", "affine_patch", "mixed_affine"):
        return 1.0+x+2.0*y
    return 0.0


def local_load(case, points, area):
    if case != "quartic":
        f = [source(case, x, y) for x, y in points]
        return [area*(sum(f)+f[i])/12.0 for i in range(3)]
    # f*N_i has degree at most 3. This rule is exact through degree 3;
    # the independent audit checks all ten reference monomials.
    rule = [([1/3, 1/3, 1/3], -27/48),
            ([3/5, 1/5, 1/5], 25/48),
            ([1/5, 3/5, 1/5], 25/48),
            ([1/5, 1/5, 3/5], 25/48)]
    result = [0.0, 0.0, 0.0]
    for weights, coefficient in rule:
        x = sum(weights[i]*points[i][0] for i in range(3))
        y = sum(weights[i]*points[i][1] for i in range(3))
        f = source(case, x, y)
        for i in range(3):
            result[i] += area*coefficient*f*weights[i]
    return result


def build_mesh(spec):
    if spec["type"] == "fan":
        center = spec["center"]
        if len(center) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 < v < 1 for v in center):
            raise ValueError("fan center must be finite and strictly inside the unit square")
        nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), tuple(center)]
        cells = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    elif spec["type"] == "grid":
        n = spec["n"]
        if type(n) is not int or not 2 <= n <= 16:
            raise ValueError("grid n must be an integer from 2 to 16")
        diagonal = spec["diagonal"]
        if diagonal not in ("forward", "alternate"):
            raise ValueError("unknown diagonal convention")
        nodes = [(i/n, j/n) for j in range(n+1) for i in range(n+1)]
        cells = []
        for j in range(n):
            for i in range(n):
                a = j*(n+1)+i
                b, d = a+1, a+n+1
                c = d+1
                if diagonal == "alternate" and (i+j)%2:
                    cells.extend([(a, b, d), (b, c, d)])
                else:
                    cells.extend([(a, b, c), (a, c, d)])
    else:
        raise ValueError("unknown mesh type")
    # Reverse alternate local orders to exercise signed geometry in normal runs.
    cells = [cell if e%2 == 0 else (cell[0], cell[2], cell[1]) for e, cell in enumerate(cells)]
    return nodes, cells


def boundary_edges(nodes, cells):
    counts = {}
    for cell in cells:
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            pair = tuple(sorted((a, b)))
            counts[pair] = counts.get(pair, 0)+1
    edges = []
    for (a, b), count in sorted(counts.items()):
        if count not in (1, 2):
            raise ValueError("nonconforming edge multiplicity")
        if count == 2:
            continue
        x0, y0 = nodes[a]
        x1, y1 = nodes[b]
        if x0 == x1 == 0:
            side, normal = "left", (-1, 0)
        elif x0 == x1 == 1:
            side, normal = "right", (1, 0)
        elif y0 == y1 == 0:
            side, normal = "bottom", (0, -1)
        elif y0 == y1 == 1:
            side, normal = "top", (0, 1)
        else:
            raise ValueError("unexpected boundary edge")
        edges.append((a, b, side, normal, math.hypot(x1-x0, y1-y0)))
    return edges


def solve_dense(A, right):
    A = [row[:] for row in A]
    right = right[:]
    n = len(right)
    for k in range(n):
        if not math.isfinite(A[k][k]) or A[k][k] <= 0:
            raise ValueError("nonpositive elimination pivot")
        for i in range(k+1, n):
            if A[i][k] == 0:
                continue
            factor = A[i][k]/A[k][k]
            for j in range(k+1, n):
                A[i][j] -= factor*A[k][j]
            right[i] -= factor*right[k]
            A[i][k] = 0.0
    answer = [0.0]*n
    for i in range(n-1, -1, -1):
        answer[i] = (right[i]-math.fsum(A[i][j]*answer[j] for j in range(i+1, n)))/A[i][i]
    return answer


def evaluate(nodes, cells, geometry, values, x, y):
    for cell, (area, gradients) in zip(cells, geometry):
        points = [nodes[i] for i in cell]
        weights = shape_values(points, gradients, x, y)
        if min(weights) >= -1e-12:
            return math.fsum(weights[i]*values[cell[i]] for i in range(3))
    raise ValueError("point outside mesh")


def solve_case(case, nodes, cells, geometry, K, edges):
    body, boundary = [0.0]*len(nodes), [0.0]*len(nodes)
    element_loads, edge_rows = [], []
    fixed = set()
    for e, (cell, (area, gradients)) in enumerate(zip(cells, geometry)):
        load = local_load(case, [nodes[i] for i in cell], area)
        for i in range(3):
            body[cell[i]] += load[i]
            element_loads.append((e, i, cell[i], load[i]))
    for a, b, side, normal, length in edges:
        is_d = case != "mixed_affine" or side == "left"
        q = 0.0 if is_d else normal[0]+2.0*normal[1]
        if is_d:
            fixed.update((a, b))
        else:
            boundary[a] += q*length/2.0
            boundary[b] += q*length/2.0
        edge_rows.append((a, b, side, length, "D" if is_d else "N", q, 0.0 if is_d else q*length/2.0))
    free = [i for i in range(len(nodes)) if i not in fixed]
    U = [0.0]*len(nodes)
    for i in fixed:
        U[i] = boundary_value(case, *nodes[i])
    right = [body[i]+boundary[i] for i in range(len(nodes))]
    reduced = [right[i]-math.fsum(K.get((i,j), 0.0)*U[j] for j in fixed) for i in free]
    A = [[K.get((i,j), 0.0) for j in free] for i in free]
    solution = solve_dense(A, reduced)
    for k, i in enumerate(free):
        U[i] = solution[k]
    residual = [math.fsum(v*U[j] for (row,j),v in K.items() if row == i)-right[i] for i in range(len(nodes))]
    return U, body, boundary, right, free, fixed, reduced, residual, element_loads, edge_rows


def run(config):
    if not config["meshes"]:
        raise ValueError("at least one mesh is required")
    if len(set(m["id"] for m in config["meshes"])) != len(config["meshes"]):
        raise ValueError("duplicate mesh id")
    if not config["cases"] or len(set(config["cases"])) != len(config["cases"]) or any(c not in CASES for c in config["cases"]):
        raise ValueError("unknown or duplicate case")
    intervals = config["profile_intervals"]
    if type(intervals) is not int or intervals < 4:
        raise ValueError("profile_intervals must be an integer at least 4")
    tables = {name: [] for name in ("geometry", "element_matrix", "matrix", "element_loads", "edges", "loads", "reduced_rhs", "nodes", "summary", "profiles", "basis", "orientations")}
    for spec in config["meshes"]:
        mesh = spec["id"]
        nodes, cells = build_mesh(spec)
        geometry = [triangle_geometry([nodes[i] for i in cell]) for cell in cells]
        if not all(math.isfinite(v) for area, g in geometry for v in [area]+[a for pair in g for a in pair]):
            raise ValueError("nonfinite triangle geometry")
        edges = boundary_edges(nodes, cells)
        K = {}
        for e, (cell, (area, gradients)) in enumerate(zip(cells, geometry)):
            points = [nodes[i] for i in cell]
            d = (points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[2][0]-points[0][0])*(points[1][1]-points[0][1])
            tables["geometry"].append((mesh,e,*cell,d,area,*[v for p in gradients for v in p]))
            local = local_matrix(area, gradients)
            for i in range(3):
                for j in range(3):
                    K[(cell[i],cell[j])] = K.get((cell[i],cell[j]), 0.0)+local[i][j]
                    tables["element_matrix"].append((mesh,e,i,j,cell[i],cell[j],local[i][j]))
        for (i,j), value in sorted(K.items()):
            tables["matrix"].append((mesh,i,j,value))
        for case in config["cases"]:
            U, body, boundary, right, free, fixed, reduced, residual, loads, edge_rows = solve_case(case,nodes,cells,geometry,K,edges)
            tables["element_loads"].extend((mesh,case,*r) for r in loads)
            tables["edges"].extend((mesh,case,*r) for r in edge_rows)
            errors = []
            for i, (x,y) in enumerate(nodes):
                ref = exact(case,x,y)
                error = None if ref is None else U[i]-ref
                if error is not None:
                    errors.append(abs(error))
                tables["nodes"].append((mesh,case,i,x,y,U[i],ref,error,residual[i],int(i in free)))
                tables["loads"].append((mesh,case,i,body[i],boundary[i],right[i],int(i in fixed),U[i] if i in fixed else None))
            for k, i in enumerate(free):
                tables["reduced_rhs"].append((mesh,case,k,i,reduced[k],K.get((i,i),0.0)))
            line_error = None
            if case == "quartic":
                line_error = 0.0
                for k in range(intervals+1):
                    x, y = k/intervals, 0.5
                    value = evaluate(nodes,cells,geometry,U,x,y)
                    ref = exact(case,x,y)
                    line_error = max(line_error,abs(value-ref))
                    tables["profiles"].append((mesh,x,y,value,ref,value-ref))
            center = evaluate(nodes,cells,geometry,U,0.5,0.5)
            tables["summary"].append((mesh,case,len(nodes),len(cells),len(free),1/spec["n"] if "n" in spec else None,center,max(abs(residual[i]) for i in free),max(abs(U[i]-boundary_value(case,*nodes[i])) for i in fixed),max(errors) if errors else None,line_error,math.fsum(area for area,g in geometry)))
        if mesh == "fan":
            phi = [0.0,0.0,0.0,0.0,1.0]
            for k in range(intervals+1):
                x = k/intervals
                tables["basis"].append((x,0.5,evaluate(nodes,cells,geometry,phi,x,0.5)))
    samples = [((0.0,0.0),(1.0,0.0),(0.0,1.0)),
               ((0.0,0.0),(2.0,0.0),(0.0,1.0)),
               ((0.0,0.0),(1.0,0.0),(0.5,0.5)),
               ((0.0,0.0),(1.0,0.0),(0.0,0.125))]
    for s, points in enumerate(samples):
        for permutation in itertools.permutations(range(3)):
            permuted = [points[i] for i in permutation]
            area, gradients = triangle_geometry(permuted)
            matrix = local_matrix(area,gradients)
            d = (permuted[1][0]-permuted[0][0])*(permuted[2][1]-permuted[0][1])-(permuted[2][0]-permuted[0][0])*(permuted[1][1]-permuted[0][1])
            tables["orientations"].append((s,*permutation,d,area,*[v for g in gradients for v in g],*[v for row in matrix for v in row]))
    for rows in tables.values():
        if any(isinstance(v,float) and not math.isfinite(v) for row in rows for v in row):
            raise ValueError("nonfinite result")
    return tables


HEADERS = {
    "geometry": "mesh,element,n0,n1,n2,d,area,g0x,g0y,g1x,g1y,g2x,g2y",
    "element_matrix": "mesh,element,i,j,row,column,value",
    "matrix": "mesh,row,column,value",
    "element_loads": "mesh,case,element,i,node,value",
    "edges": "mesh,case,a,b,side,length,type,q,increment",
    "loads": "mesh,case,node,body,boundary,right,fixed,value",
    "reduced_rhs": "mesh,case,index,node,right,diagonal",
    "nodes": "mesh,case,node,x,y,solution,exact,error,residual,free",
    "summary": "mesh,case,nodes,triangles,free,h,center,tested_residual,boundary_error,nodal_error,line_error,total_area",
    "profiles": "mesh,x,y,solution,exact,error",
    "basis": "x,y,value",
    "orientations": "sample,p0,p1,p2,d,area,g0x,g0y,g1x,g1y,g2x,g2y,k00,k01,k02,k10,k11,k12,k20,k21,k22"
}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root/"config/problems.json")
    parser.add_argument("--output", type=Path, default=root/"results")
    args = parser.parse_args()
    tables = run(json.loads(args.config.read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    for name, rows in tables.items():
        with (args.output/(name+".csv")).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.writer(stream,lineterminator="\n")
            writer.writerow(HEADERS[name].split(","))
            writer.writerows(rows)
    summary = tables["summary"]
    log = ["Python "+platform.python_version(),"core dependencies: standard library", "quadrature: exact through degree 3 for f*N_i", "cases solved: "+str(len(summary)), "maximum tested residual: "+format(max(r[7] for r in summary),".17g")]
    log.extend(name+" rows: "+str(len(rows)) for name,rows in tables.items())
    (args.output/"run.txt").write_text("\n".join(log)+"\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
