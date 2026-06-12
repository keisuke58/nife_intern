"""Mechanically-closed biofilm material model: NSP composition -> growth eigenstrain -> stress.

This closes the constitutive law so it returns a real Cauchy stress at each Gauss point — the
object the FEM (Abaqus UMAT / ANSYS USERMAT) actually needs.

Mechanism (one assumed coupling; the modelling choice the thesis would justify/extend):
  1. The NSP state g evolves the local community composition (reaction; one implicit step).
  2. A growth/swelling **eigenstrain** is driven by the P. gingivalis fraction phi_Pg:
         eps_growth = beta * phi_Pg          (isotropic volumetric)
     i.e. dysbiotic Pg overgrowth swells the matrix. (Swap for any phi-functional later.)
  3. Stress from linear elasticity on the *mechanical* (elastic) strain only:
         sigma = C : (eps_total - eps_growth * I_vol)
     DDSDDE = C  (eigenstrain is independent of the current mechanical strain, so the mechanical
     tangent is the elastic tangent; the composition coupling enters through STATEV history).

State (STATEV / SDV): the 12-dim NSP state g = [phi_1..5, phi_0, psi_1..5, gamma].
Contract matches material_model.py so the UMAT bridge is unchanged:
    stress(6), ddsdde(6,6), statev = evaluate(strain, dstrain, statev, props)
props: {"E","nu","beta", "cond" (only used to init), "psi_init"}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import nsp_material_model as nm
from material_model import elastic_tangent

PG_INDEX = 4                       # species order: So,An,Vd/Vp,Fn,Pg
I_VOL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # volumetric (isotropic swelling), Voigt
I_Z = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])     # substratum-normal (depth) growth, Voigt
NSTATEV = nm.N_STATE               # 12

# CLSM-calibrated growth coefficients (see calibrate_beta.py); fall back to a placeholder.
_CALIB_FILE = Path(__file__).resolve().parent / 'beta_calibration.json'


def _calib(cond: str):
    """Return (beta_depth, intercept) from CLSM calibration, or a placeholder."""
    try:
        d = json.loads(_CALIB_FILE.read_text())[cond]
        return d["beta_depth"], d["intercept"]
    except Exception:
        return 17.0, -1.34  # DH placeholder if calibration json absent

# one NSP evaluator per condition (cached)
_EVALS: dict = {}


def _nsp_step_for(cond: str):
    if cond not in _EVALS:
        _EVALS[cond] = nm.make_evaluator(nm.load_params(cond))[0]  # step closure
    return _EVALS[cond]


def init_statev(phi, psi_init: float = 0.999) -> np.ndarray:
    """Valid initial 12-dim NSP STATEV from a 5-vector composition."""
    return np.asarray(nm.make_initial_state(phi, psi_init=psi_init), float)


def evaluate(strain, dstrain, statev, props):
    strain = np.asarray(strain, float)
    dstrain = np.asarray(dstrain, float)
    cond = props.get("cond", "DH")
    beta = props.get("beta", 0.05)

    # zero/empty STATEV -> initialise NSP state (Abaqus *DEPVAR starts at 0)
    statev = np.asarray(statev, float) if statev is not None else np.zeros(NSTATEV)
    if statev.size != NSTATEV or not np.any(statev):
        statev = init_statev(props.get("phi_init", np.full(5, 0.2)),
                             props.get("psi_init", 0.999))

    # 1. advance composition (reaction)
    step = _nsp_step_for(cond)
    g_new = np.asarray(step(statev), float)
    phi_pg = g_new[PG_INDEX]

    # 2. growth eigenstrain from Pg fraction
    mode = props.get("growth_mode", "depth")   # data-grounded default: anisotropic depth growth
    if mode == "depth":
        # CLSM: lateral FOV fixed, biofilm thickens in z -> eigenstrain in zz only.
        # eps_zz = beta_depth*phi_Pg + intercept (small-strain linearisation; clip >=0).
        beta_d, intcpt = (beta, 0.0) if "beta" in props and props.get("override_calib") \
            else _calib(cond)
        eps_zz = max(0.0, beta_d * phi_pg + intcpt)
        eps_growth_vec = eps_zz * I_Z
    else:  # "isotropic" volumetric swelling (comparison / placeholder)
        eps_growth_vec = (beta * phi_pg) * I_VOL

    # 3. elastic stress on mechanical strain
    C = elastic_tangent(props["E"], props["nu"])
    eps_mech = (strain + dstrain) - eps_growth_vec
    stress = C @ eps_mech
    ddsdde = C
    return stress, ddsdde, g_new
