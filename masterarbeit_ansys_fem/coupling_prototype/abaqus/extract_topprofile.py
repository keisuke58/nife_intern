import sys
from odbAccess import openOdb
J = sys.argv[1]; Nx = 48; Nz = 8; L = 8.0
def nid(i, j, k):
    return 1 + i + (Nx + 1) * (j + 2 * k)
o = openOdb(J + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
U = {v.nodeLabel: v.data[2] for v in fr.fieldOutputs['U'].values}
with open(J + "_topprofile.csv", "w") as f:
    f.write("x,u3\n")
    for i in range(Nx + 1):
        f.write("%.4f,%.6e\n" % (i * L / Nx, U.get(nid(i, 0, Nz), 0.0)))
print(J, "top profile written")
o.close()
