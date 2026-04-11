from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class HexMesh:
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    @property
    def hx(self) -> float:
        return self.lx / self.nx

    @property
    def hy(self) -> float:
        return self.ly / self.ny

    @property
    def hz(self) -> float:
        return self.lz / self.nz

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def n_elems(self) -> int:
        return self.nx * self.ny * self.nz

    def node_id(self, ix: int, iy: int, iz: int) -> int:
        return ix + (self.nx + 1) * (iy + (self.ny + 1) * iz)

    def iter_elements(self):
        for ez in range(self.nz):
            for ey in range(self.ny):
                for ex in range(self.nx):
                    n000 = self.node_id(ex, ey, ez)
                    n100 = self.node_id(ex + 1, ey, ez)
                    n010 = self.node_id(ex, ey + 1, ez)
                    n110 = self.node_id(ex + 1, ey + 1, ez)
                    n001 = self.node_id(ex, ey, ez + 1)
                    n101 = self.node_id(ex + 1, ey, ez + 1)
                    n011 = self.node_id(ex, ey + 1, ez + 1)
                    n111 = self.node_id(ex + 1, ey + 1, ez + 1)
                    yield (n000, n100, n010, n110, n001, n101, n011, n111)

    def top_dirichlet_nodes(self) -> np.ndarray:
        iz = self.nz
        ids = []
        for iy in range(self.ny + 1):
            for ix in range(self.nx + 1):
                ids.append(self.node_id(ix, iy, iz))
        return np.array(ids, dtype=int)

    def bottom_surface_nodes(self) -> np.ndarray:
        iz = 0
        ids = []
        for iy in range(self.ny + 1):
            for ix in range(self.nx + 1):
                ids.append(self.node_id(ix, iy, iz))
        return np.array(ids, dtype=int)


def _hex_q1_element_matrices(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w = np.array([1.0, 1.0])

    det_j = (hx * hy * hz) / 8.0
    sx = 2.0 / hx
    sy = 2.0 / hy
    sz = 2.0 / hz

    m = np.zeros((8, 8), dtype=float)
    k = np.zeros((8, 8), dtype=float)

    signs = np.array(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (-1, 1, -1),
            (1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, 1, 1),
        ],
        dtype=float,
    )
    sxi = signs[:, 0]
    seta = signs[:, 1]
    szeta = signs[:, 2]

    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            for l, zeta in enumerate(gp):
                weight = w[i] * w[j] * w[l] * det_j

                n = 0.125 * (1.0 + sxi * xi) * (1.0 + seta * eta) * (1.0 + szeta * zeta)
                dndxi = 0.125 * sxi * (1.0 + seta * eta) * (1.0 + szeta * zeta)
                dndeta = 0.125 * (1.0 + sxi * xi) * seta * (1.0 + szeta * zeta)
                dndzeta = 0.125 * (1.0 + sxi * xi) * (1.0 + seta * eta) * szeta

                gx = dndxi * sx
                gy = dndeta * sy
                gz = dndzeta * sz

                m += weight * np.outer(n, n)
                k += weight * (np.outer(gx, gx) + np.outer(gy, gy) + np.outer(gz, gz))

    return m, k


def assemble_mass_stiffness(mesh: HexMesh) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    me, ke = _hex_q1_element_matrices(mesh.hx, mesh.hy, mesh.hz)

    rows: list[int] = []
    cols: list[int] = []
    m_data: list[float] = []
    k_data: list[float] = []

    for elem in mesh.iter_elements():
        for a_local, a in enumerate(elem):
            for b_local, b in enumerate(elem):
                rows.append(a)
                cols.append(b)
                m_data.append(me[a_local, b_local])
                k_data.append(ke[a_local, b_local])

    n = mesh.n_nodes
    m = sp.coo_matrix((m_data, (rows, cols)), shape=(n, n)).tocsr()
    k = sp.coo_matrix((k_data, (rows, cols)), shape=(n, n)).tocsr()
    return m, k


def _apply_dirichlet(a: sp.csr_matrix, b: np.ndarray, fixed: np.ndarray, values: float | np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    if np.isscalar(values):
        vals = np.full_like(fixed, float(values), dtype=float)
    else:
        vals = np.asarray(values, dtype=float)
        if vals.shape != fixed.shape:
            raise ValueError("Dirichlet values shape mismatch")

    a_lil = a.tolil(copy=True)
    for idx, v in zip(fixed, vals):
        a_lil.rows[idx] = [idx]
        a_lil.data[idx] = [1.0]
        b[idx] = float(v)
    return a_lil.tocsr(), b


@dataclass(frozen=True)
class FemPgParams:
    d_nutrient: float = 0.05
    k_consume: float = 0.15
    mu_phi: float = 0.25      # Pg µ_max ~0.2-0.4 /h (literature)
    km_phi: float = 0.2
    k_detach: float = 0.04    # moderate detachment to prevent saturation
    d_phi: float = 0.0002
    c_top: float = 1.0
    phi_seed: float = 0.02
    seed_fraction: float = 0.05
    nh4_init: float = 12.0
    k_nh4: float = 1500.0
    km_nh4: float = 5.0          # NH4 half-saturation (mM); nh4_init/km_nh4 ~ 2.4 → strong depletion effect
    phi_cov_thresh: float = 0.01


@dataclass(frozen=True)
class SurfaceParams:
    name: str
    phi_seed: float
    k_detach: float
    km_nh4: float = 5.0


def run_fem_pg_single(
    mesh: HexMesh,
    params: FemPgParams,
    t_end_h: float = 24.0 * 6,
    dt_h: float = 1.0,
    dhna_mu_scale: float = 1.2,
    rng_seed: int = 0,
):
    m, k = assemble_mass_stiffness(mesh)
    m_lump = np.asarray(m.sum(axis=1)).ravel()
    m_lump[m_lump == 0.0] = 1.0

    top_nodes = mesh.top_dirichlet_nodes()
    bottom_nodes = mesh.bottom_surface_nodes()

    n = mesh.n_nodes
    rng = np.random.default_rng(rng_seed)

    c0 = np.full(n, params.c_top, dtype=float)
    phi0 = np.zeros(n, dtype=float)
    seeded = rng.random(bottom_nodes.size) < float(params.seed_fraction)
    vals = params.phi_seed * (0.5 + rng.random(bottom_nodes.size))
    phi0[bottom_nodes] = np.where(seeded, vals, 0.0)
    phi0 = np.clip(phi0, 0.0, 1.0)

    def simulate(mu_scale: float):
        c = c0.copy()
        phi = phi0.copy()

        steps = int(np.ceil(t_end_h / dt_h))
        times = np.arange(steps + 1) * dt_h
        pg_volume = np.zeros(steps + 1, dtype=float)
        area_mean = np.zeros(steps + 1, dtype=float)
        area_cov = np.zeros(steps + 1, dtype=float)
        nh4 = np.zeros(steps + 1, dtype=float)

        nh4_cur = float(params.nh4_init)

        pg_volume[0] = float(m_lump @ phi)
        area_mean[0] = float(phi[bottom_nodes].mean())
        area_cov[0] = float((phi[bottom_nodes] > params.phi_cov_thresh).mean())
        nh4[0] = nh4_cur

        a_base = m + (dt_h * params.d_nutrient) * k

        a_phi = m + (dt_h * params.d_phi) * k
        lu_phi = spla.factorized(a_phi.tocsc())

        for step in range(steps):
            r_diag = dt_h * params.k_consume * m_lump * phi
            a = a_base + sp.diags(r_diag, offsets=0, format="csr")
            rhs = m @ c
            a, rhs = _apply_dirichlet(a, rhs, top_nodes, params.c_top)
            c = spla.spsolve(a.tocsc(), rhs)
            c = np.clip(c, 0.0, None)

            nh4_monod = nh4_cur / (params.km_nh4 + nh4_cur + 1e-15)
            growth = params.mu_phi * mu_scale * phi * (1.0 - phi) * (c / (params.km_phi + c + 1e-15)) * nh4_monod
            rhs_phi = m @ (phi + dt_h * (growth - params.k_detach * phi))
            phi = lu_phi(rhs_phi)
            phi = np.clip(phi, 0.0, 1.0)

            total_growth = float(m_lump @ np.maximum(growth, 0.0))
            nh4_cur = max(0.0, nh4_cur - dt_h * params.k_nh4 * total_growth)

            pg_volume[step + 1] = float(m_lump @ phi)
            area_mean[step + 1] = float(phi[bottom_nodes].mean())
            area_cov[step + 1] = float((phi[bottom_nodes] > params.phi_cov_thresh).mean())
            nh4[step + 1] = nh4_cur

        return dict(time_h=times, volume=pg_volume, area_mean=area_mean, area_cov=area_cov, nh4=nh4)

    baseline = simulate(mu_scale=1.0)
    dhna = simulate(mu_scale=dhna_mu_scale)
    return baseline, dhna


def run_fem_pg_surfaces(
    mesh: HexMesh,
    base_params: FemPgParams,
    surfaces: list[SurfaceParams],
    t_end_h: float,
    dt_h: float,
    dhna_mu_scale: float,
    rng_seed: int = 0,
):
    results = {}
    for s in surfaces:
        p = FemPgParams(
            d_nutrient=base_params.d_nutrient,
            k_consume=base_params.k_consume,
            mu_phi=base_params.mu_phi,
            km_phi=base_params.km_phi,
            k_detach=s.k_detach,
            d_phi=base_params.d_phi,
            c_top=base_params.c_top,
            phi_seed=s.phi_seed,
            seed_fraction=base_params.seed_fraction,
            nh4_init=base_params.nh4_init,
            k_nh4=base_params.k_nh4,
            km_nh4=s.km_nh4,
            phi_cov_thresh=base_params.phi_cov_thresh,
        )
        baseline, dhna = run_fem_pg_single(
            mesh=mesh,
            params=p,
            t_end_h=t_end_h,
            dt_h=dt_h,
            dhna_mu_scale=dhna_mu_scale,
            rng_seed=rng_seed,
        )
        results[s.name] = dict(baseline=baseline, dhna=dhna, surface=s, params=p)
    return results


def calibrate_surface_params(
    mesh: HexMesh,
    base_params: FemPgParams,
    target_baseline_area_day6: float,
    target_dhna_area_day6: float,
    t_end_h: float,
    dt_h: float,
    dhna_mu_scale: float,
    phi_seed_grid: np.ndarray,
    k_detach_grid: np.ndarray,
    rng_seed: int = 0,
):
    best = None
    best_err = float("inf")

    for phi_seed in phi_seed_grid:
        for k_detach in k_detach_grid:
            p = FemPgParams(
                d_nutrient=base_params.d_nutrient,
                k_consume=base_params.k_consume,
                mu_phi=base_params.mu_phi,
                km_phi=base_params.km_phi,
                k_detach=float(k_detach),
                d_phi=base_params.d_phi,
                c_top=base_params.c_top,
                phi_seed=float(phi_seed),
                seed_fraction=base_params.seed_fraction,
                nh4_init=base_params.nh4_init,
                k_nh4=base_params.k_nh4,
                km_nh4=base_params.km_nh4,
                phi_cov_thresh=base_params.phi_cov_thresh,
            )
            baseline, dhna = run_fem_pg_single(
                mesh=mesh,
                params=p,
                t_end_h=t_end_h,
                dt_h=dt_h,
                dhna_mu_scale=dhna_mu_scale,
                rng_seed=rng_seed,
            )
            b = float(baseline["area_cov"][-1])
            d = float(dhna["area_cov"][-1])
            err = (b - target_baseline_area_day6) ** 2 + (d - target_dhna_area_day6) ** 2
            if err < best_err:
                best_err = err
                best = dict(
                    phi_seed=float(phi_seed),
                    k_detach=float(k_detach),
                    area_baseline=b,
                    area_dhna=d,
                )

    if best is None:
        raise RuntimeError("Calibration failed to find any candidate.")
    return best, best_err


def _grid_search_surface(
    mesh: HexMesh,
    base_params: FemPgParams,
    targets: dict,
    t_end_h: float,
    dt_h: float,
    dhna_mu_scale: float,
    phi_seed_grid: np.ndarray,
    k_detach_grid: np.ndarray,
    km_nh4_grid: np.ndarray,
    rng_seed: int = 0,
    verbose: bool = False,
) -> tuple["SurfaceParams", float, float]:
    """Per-surface grid search (phi_seed × k_detach × km_nh4) minimising volume RMSE."""
    days_h = np.asarray(targets["days_h"], dtype=float)
    target_concat = np.concatenate([
        np.asarray(targets["baseline"], dtype=float),
        np.asarray(targets["dhna"], dtype=float),
    ])

    best_sp: "SurfaceParams | None" = None
    best_scale = 1.0
    best_rmse = float("inf")

    for km_nh4 in km_nh4_grid:
        for phi_seed in phi_seed_grid:
            for k_detach in k_detach_grid:
                p = FemPgParams(
                    d_nutrient=base_params.d_nutrient,
                    k_consume=base_params.k_consume,
                    mu_phi=base_params.mu_phi,
                    km_phi=base_params.km_phi,
                    k_detach=float(k_detach),
                    d_phi=base_params.d_phi,
                    c_top=base_params.c_top,
                    phi_seed=float(phi_seed),
                    seed_fraction=base_params.seed_fraction,
                    nh4_init=base_params.nh4_init,
                    k_nh4=base_params.k_nh4,
                    km_nh4=float(km_nh4),
                    phi_cov_thresh=base_params.phi_cov_thresh,
                )
                baseline, dhna = run_fem_pg_single(
                    mesh=mesh, params=p, t_end_h=t_end_h, dt_h=dt_h,
                    dhna_mu_scale=dhna_mu_scale, rng_seed=rng_seed,
                )
                t_sim = baseline["time_h"]
                y = np.concatenate([
                    np.interp(days_h, t_sim, np.asarray(baseline["volume"])),
                    np.interp(days_h, t_sim, np.asarray(dhna["volume"])),
                ])
                denom = float(np.dot(y, y))
                scale = float(np.dot(y, target_concat) / denom) if denom > 0 else 1.0
                # reject degenerate scales (volume effectively zero)
                if scale > 1e4 or scale <= 0:
                    continue
                rmse = float(np.sqrt(np.mean((scale * y - target_concat) ** 2)))
                if verbose:
                    print(f"  km_nh4={km_nh4:.2f} phi_seed={phi_seed:.4f} k_detach={k_detach:.4f}  scale={scale:.3f}  RMSE={rmse:.4f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_scale = scale
                    best_sp = SurfaceParams(name="", phi_seed=float(phi_seed), k_detach=float(k_detach), km_nh4=float(km_nh4))

    if best_sp is None:
        raise RuntimeError("Grid search produced no valid candidates (all scales degenerate).")
    return best_sp, best_scale, best_rmse


def _fit_scale(sim_time_h: np.ndarray, sim_vol: np.ndarray, day_points: np.ndarray, target_vol: np.ndarray) -> float:
    y = np.interp(day_points, sim_time_h, sim_vol)
    denom = float(np.dot(y, y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(y, target_vol) / denom)


def fit_k_nh4(
    mesh: HexMesh,
    base_params: FemPgParams,
    target_nh4_day6: float,
    t_end_h: float,
    dt_h: float,
    k_lo: float = 10.0,
    k_hi: float = 1e6,
    tol: float = 0.1,
    max_iter: int = 40,
) -> float:
    """Bisection on k_nh4 so that baseline NH4(day6) == target_nh4_day6."""
    def nh4_at_day6(k_nh4: float) -> float:
        p = FemPgParams(
            d_nutrient=base_params.d_nutrient,
            k_consume=base_params.k_consume,
            mu_phi=base_params.mu_phi,
            km_phi=base_params.km_phi,
            k_detach=base_params.k_detach,
            d_phi=base_params.d_phi,
            c_top=base_params.c_top,
            phi_seed=base_params.phi_seed,
            seed_fraction=base_params.seed_fraction,
            nh4_init=base_params.nh4_init,
            k_nh4=k_nh4,
            km_nh4=base_params.km_nh4,
            phi_cov_thresh=base_params.phi_cov_thresh,
        )
        baseline, _ = run_fem_pg_single(mesh=mesh, params=p, t_end_h=t_end_h, dt_h=dt_h)
        return float(baseline["nh4"][-1])

    f_lo = nh4_at_day6(k_lo) - target_nh4_day6
    f_hi = nh4_at_day6(k_hi) - target_nh4_day6
    if f_lo * f_hi > 0:
        # fallback: return the end with smaller abs error
        print(f"[fit_k_nh4] bracket failed: f_lo={f_lo:.3f} f_hi={f_hi:.3f}; returning best guess")
        return k_lo if abs(f_lo) < abs(f_hi) else k_hi

    for _ in range(max_iter):
        k_mid = 0.5 * (k_lo + k_hi)
        f_mid = nh4_at_day6(k_mid) - target_nh4_day6
        if abs(f_mid) < tol:
            return k_mid
        if f_lo * f_mid < 0:
            k_hi = k_mid
            f_hi = f_mid
        else:
            k_lo = k_mid
            f_lo = f_mid
    return 0.5 * (k_lo + k_hi)


def _plot_single(baseline: dict, dhna: dict, outpath: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = baseline["time_h"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FEM prototype (Q1 hex, Galerkin): nutrient diffusion + biofilm growth", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(t, baseline["volume"], color="#555555", lw=2, label="Baseline")
    ax.plot(t, dhna["volume"], color="#9C27B0", lw=2, ls="--", label="DHNA proxy")
    ax.set_title("Biofilm volume proxy")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("∫φ dV (arb.)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t, baseline["area_cov"], color="#555555", lw=2, label="Baseline (cov)")
    ax.plot(t, dhna["area_cov"], color="#9C27B0", lw=2, ls="--", label="DHNA (cov)")
    ax.plot(t, baseline["area_mean"], color="#555555", lw=1, ls=":", label="Baseline (mean)")
    ax.plot(t, dhna["area_mean"], color="#9C27B0", lw=1, ls=":", label="DHNA (mean)")
    ax.set_title("Area fraction (bottom)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t, baseline["nh4"], color="#555555", lw=2, label="Baseline")
    ax.plot(t, dhna["nh4"], color="#9C27B0", lw=2, ls="--", label="DHNA proxy")
    ax.set_title("NH4 proxy")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("NH4 (arb.)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_surfaces(results: dict, outpath: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = list(results.keys())
    fig, axes = plt.subplots(len(surfaces), 3, figsize=(15, 4 * len(surfaces)))
    if len(surfaces) == 1:
        axes = np.array([axes])

    fig.suptitle("FEM prototype (Q1 hex, Galerkin): surface roughness proxy", fontsize=11, fontweight="bold")

    for row, name in enumerate(surfaces):
        baseline = results[name]["baseline"]
        dhna = results[name]["dhna"]
        s = results[name]["surface"]
        t = baseline["time_h"]

        ax = axes[row, 0]
        ax.plot(t, baseline["volume"], color="#555555", lw=2, label="Baseline")
        ax.plot(t, dhna["volume"], color="#9C27B0", lw=2, ls="--", label="DHNA proxy")
        ax.set_title(f"{name}: volume proxy")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("∫φ dV (arb.)")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.plot(t, baseline["area_cov"], color="#555555", lw=2, label="Baseline (cov)")
        ax.plot(t, dhna["area_cov"], color="#9C27B0", lw=2, ls="--", label="DHNA (cov)")
        ax.plot(t, baseline["area_mean"], color="#555555", lw=1, ls=":", label="Baseline (mean)")
        ax.plot(t, dhna["area_mean"], color="#9C27B0", lw=1, ls=":", label="DHNA (mean)")
        ax.set_title(f"{name}: area fraction (bottom)  seed={s.phi_seed:.3f}, k_detach={s.k_detach:.4f}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        ax.plot(t, baseline["nh4"], color="#555555", lw=2, label="Baseline")
        ax.plot(t, dhna["nh4"], color="#9C27B0", lw=2, ls="--", label="DHNA proxy")
        ax.set_title(f"{name}: NH4 proxy")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("NH4 (arb.)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--nz", type=int, default=6)
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument("--lz", type=float, default=1.0)
    ap.add_argument("--dt_h", type=float, default=1.0)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--dhna_mu_scale", type=float, default=1.2)
    ap.add_argument("--out", type=Path, default=Path("/home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/fem_pg_demo.png"))
    ap.add_argument("--roughness", action="store_true")
    ap.add_argument("--calibrate_roughness", action="store_true")
    ap.add_argument("--fit_fig2", action="store_true")
    args = ap.parse_args(argv)

    mesh = HexMesh(nx=args.nx, ny=args.ny, nz=args.nz, lx=args.lx, ly=args.ly, lz=args.lz)
    t_end_h = 24.0 * args.days
    base_params = FemPgParams()

    if args.calibrate_roughness:
        target = {
            "rougher": {"baseline": 0.185, "dhna": 0.315},
            "smoother": {"baseline": 0.166, "dhna": 0.260},
        }
        phi_seed_grid = np.linspace(0.001, 0.030, 6)
        k_detach_grid = np.linspace(0.000, 0.120, 9)

        best_params = {}
        for name, tgt in target.items():
            best, err = calibrate_surface_params(
                mesh=mesh,
                base_params=base_params,
                target_baseline_area_day6=tgt["baseline"],
                target_dhna_area_day6=tgt["dhna"],
                t_end_h=t_end_h,
                dt_h=args.dt_h,
                dhna_mu_scale=args.dhna_mu_scale,
                phi_seed_grid=phi_seed_grid,
                k_detach_grid=k_detach_grid,
            )
            best_params[name] = dict(best=best, err=err, target=tgt)
            print(name, "best", best, "err", err)

        surfaces = [
            SurfaceParams("rougher", phi_seed=best_params["rougher"]["best"]["phi_seed"], k_detach=best_params["rougher"]["best"]["k_detach"]),
            SurfaceParams("smoother", phi_seed=best_params["smoother"]["best"]["phi_seed"], k_detach=best_params["smoother"]["best"]["k_detach"]),
        ]
        results = run_fem_pg_surfaces(
            mesh=mesh,
            base_params=base_params,
            surfaces=surfaces,
            t_end_h=t_end_h,
            dt_h=args.dt_h,
            dhna_mu_scale=args.dhna_mu_scale,
        )
        _plot_surfaces(results, args.out)
        print(args.out)
        return 0

    if args.roughness:
        surfaces = [
            SurfaceParams("rougher", phi_seed=0.030, k_detach=0.004),
            SurfaceParams("smoother", phi_seed=0.018, k_detach=0.010),
        ]
        results = run_fem_pg_surfaces(
            mesh=mesh,
            base_params=base_params,
            surfaces=surfaces,
            t_end_h=t_end_h,
            dt_h=args.dt_h,
            dhna_mu_scale=args.dhna_mu_scale,
        )
        _plot_surfaces(results, args.out)
        print(args.out)
        return 0

    if args.fit_fig2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Paper data from Mukherjee 2025 Fig 2d/e (digitized, µm³/µm²) ---
        # time points: 1d, 2d, 3d, 4d, 6d  (hours: 24,48,72,96,144)
        targets = {
            "rougher": {
                "days_h": np.array([24, 48, 72, 96, 144], dtype=float),
                "baseline": np.array([0.644, 1.127, 1.317, 0.838, 0.890], dtype=float),
                "dhna":     np.array([0.553, 0.415, 0.432, 1.330, 0.648], dtype=float),
            },
            "smoother": {
                "days_h": np.array([24, 48, 72, 96, 144], dtype=float),
                "baseline": np.array([0.238, 0.380, 0.384, 0.229, 0.216], dtype=float),
                "dhna":     np.array([0.324, 0.402, 0.341, 0.194, 0.276], dtype=float),
            },
        }
        # Fig 1 NH4: baseline at day6 ~5 mM, DHNA depletes faster (~0 by day4)
        nh4_target_baseline_day6 = 5.0   # mM (from Mukherjee 2025 Fig 1)

        # --- Step 1: fit k_nh4 so baseline NH4(day6) = 5 mM ---
        print("[fit_fig2] Step 1: fitting k_nh4 for NH4(day6) = 5 mM ...")
        k_nh4_fitted = fit_k_nh4(
            mesh=mesh,
            base_params=base_params,
            target_nh4_day6=nh4_target_baseline_day6,
            t_end_h=t_end_h,
            dt_h=args.dt_h,
            k_lo=0.1,
            k_hi=1e5,
        )
        print(f"[fit_fig2] k_nh4 = {k_nh4_fitted:.1f}")
        fitted_params = FemPgParams(
            d_nutrient=base_params.d_nutrient,
            k_consume=base_params.k_consume,
            mu_phi=base_params.mu_phi,
            km_phi=base_params.km_phi,
            k_detach=base_params.k_detach,
            d_phi=base_params.d_phi,
            c_top=base_params.c_top,
            phi_seed=base_params.phi_seed,
            seed_fraction=base_params.seed_fraction,
            nh4_init=base_params.nh4_init,
            k_nh4=k_nh4_fitted,
            km_nh4=base_params.km_nh4,
            phi_cov_thresh=base_params.phi_cov_thresh,
        )

        # --- Step 2: per-surface grid search for phi_seed × k_detach × km_nh4 ---
        # rougher: refine around best (phi_seed=0.030, k_detach=0.020, km_nh4=0.5)
        # smoother: push phi_seed up to recover volume (fix scale explosion)
        surface_grids = {
            "rougher": dict(
                phi_seed_grid = np.array([0.020, 0.025, 0.030, 0.040, 0.050]),
                k_detach_grid = np.array([0.005, 0.010, 0.015, 0.020, 0.030, 0.040]),
                km_nh4_grid   = np.array([0.3, 0.5, 0.8, 1.5, 3.0]),
            ),
            "smoother": dict(
                phi_seed_grid = np.array([0.010, 0.020, 0.030, 0.050, 0.070]),
                k_detach_grid = np.array([0.005, 0.010, 0.020, 0.040, 0.080]),
                km_nh4_grid   = np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
            ),
        }
        best_sp: dict[str, SurfaceParams] = {}
        scales: dict[str, float] = {}
        for name in ("rougher", "smoother"):
            g = surface_grids[name]
            phi_seed_grid = g["phi_seed_grid"]
            k_detach_grid = g["k_detach_grid"]
            km_nh4_grid   = g["km_nh4_grid"]
            n_total = len(phi_seed_grid) * len(k_detach_grid) * len(km_nh4_grid)
            print(f"[fit_fig2] Step 2 [{name}]: {n_total} combos ...")
            sp, scale, rmse = _grid_search_surface(
                mesh=mesh,
                base_params=fitted_params,
                targets=targets[name],
                t_end_h=t_end_h,
                dt_h=args.dt_h,
                dhna_mu_scale=args.dhna_mu_scale,
                phi_seed_grid=phi_seed_grid,
                k_detach_grid=k_detach_grid,
                km_nh4_grid=km_nh4_grid,
                verbose=True,
            )
            sp = SurfaceParams(name=name, phi_seed=sp.phi_seed, k_detach=sp.k_detach, km_nh4=sp.km_nh4)
            best_sp[name] = sp
            scales[name] = scale
            print(f"[fit_fig2] {name}: phi_seed={sp.phi_seed:.4f}, k_detach={sp.k_detach:.4f}, km_nh4={sp.km_nh4:.2f}  scale={scale:.4f}  RMSE={rmse:.4f}")

        # --- Step 3: final run with best per-surface params ---
        surfaces = [best_sp["rougher"], best_sp["smoother"]]
        results = run_fem_pg_surfaces(
            mesh=mesh,
            base_params=fitted_params,
            surfaces=surfaces,
            t_end_h=t_end_h,
            dt_h=args.dt_h,
            dhna_mu_scale=args.dhna_mu_scale,
        )

        # --- Step 4: compute RMSE ---
        for name in ("rougher", "smoother"):
            t_sim = results[name]["baseline"]["time_h"]
            s = scales[name]
            days_h = targets[name]["days_h"]
            pred_b = s * np.interp(days_h, t_sim, results[name]["baseline"]["volume"])
            pred_d = s * np.interp(days_h, t_sim, results[name]["dhna"]["volume"])
            rmse_b = float(np.sqrt(np.mean((pred_b - targets[name]["baseline"])**2)))
            rmse_d = float(np.sqrt(np.mean((pred_d - targets[name]["dhna"])**2)))
            print(f"[fit_fig2] {name}  RMSE baseline={rmse_b:.4f}  DHNA={rmse_d:.4f}  µm³/µm²")

        # --- Step 5: plot with paper data overlay ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        sp_info = "  ".join(
            f"{n}: seed={best_sp[n].phi_seed:.3f}, kd={best_sp[n].k_detach:.3f}, km_nh4={best_sp[n].km_nh4:.1f}"
            for n in ("rougher", "smoother")
        )
        fig.suptitle(
            f"FEM vs Mukherjee 2025 (Fig 2 volume, Fig 1 NH4)\n"
            f"k_nh4={k_nh4_fitted:.0f}  dhna_mu_scale={args.dhna_mu_scale}\n{sp_info}",
            fontsize=10, fontweight="bold",
        )

        for row, name in enumerate(("rougher", "smoother")):
            t_sim = results[name]["baseline"]["time_h"]
            s = scales[name]
            days_h = targets[name]["days_h"]

            # Panel 0: volume + paper data
            ax = axes[row, 0]
            ax.plot(t_sim, s * np.array(results[name]["baseline"]["volume"]),
                    color="#555555", lw=2, label="Sim baseline")
            ax.plot(t_sim, s * np.array(results[name]["dhna"]["volume"]),
                    color="#9C27B0", lw=2, ls="--", label="Sim DHNA")
            ax.scatter(days_h, targets[name]["baseline"],
                       color="#555555", marker="o", s=50, zorder=5, label="Paper baseline")
            ax.scatter(days_h, targets[name]["dhna"],
                       color="#9C27B0", marker="^", s=50, zorder=5, label="Paper DHNA")
            ax.set_title(f"{name}: biofilm volume (µm³/µm²)")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Volume (µm³/µm²)")
            ax.legend(fontsize=7)

            # Panel 1: area fraction
            ax = axes[row, 1]
            ax.plot(t_sim, results[name]["baseline"]["area_cov"],
                    color="#555555", lw=2, label="Baseline (cov)")
            ax.plot(t_sim, results[name]["dhna"]["area_cov"],
                    color="#9C27B0", lw=2, ls="--", label="DHNA (cov)")
            ax.set_ylim(0, 1)
            ax.set_title(f"{name}: area coverage")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Fraction")
            ax.legend(fontsize=7)

            # Panel 2: NH4
            ax = axes[row, 2]
            ax.plot(t_sim, results[name]["baseline"]["nh4"],
                    color="#555555", lw=2, label="Sim baseline")
            ax.plot(t_sim, results[name]["dhna"]["nh4"],
                    color="#9C27B0", lw=2, ls="--", label="Sim DHNA")
            ax.axhline(nh4_target_baseline_day6, color="gray", ls=":", lw=1,
                       label=f"Paper NH4(day6)={nh4_target_baseline_day6} mM")
            ax.set_title(f"{name}: NH4 (mM)")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("NH4 (mM)")
            ax.legend(fontsize=7)

        plt.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(args.out)
        return 0

    baseline, dhna = run_fem_pg_single(
        mesh=mesh,
        params=base_params,
        t_end_h=t_end_h,
        dt_h=args.dt_h,
        dhna_mu_scale=args.dhna_mu_scale,
    )
    _plot_single(baseline, dhna, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
