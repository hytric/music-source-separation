"""Reuse target projections without changing the established metric definitions."""
from __future__ import annotations

import numpy as np


def score_sets(targets, sets):
    from experiments.banelp.metrics import _db_ratio, SILENT_TARGET_ENERGY
    from banelp_refine.common import SOURCES
    S = np.asarray(targets, dtype=np.float64).reshape(4, -1).T
    names = list(sets)
    Z = np.concatenate([np.asarray(sets[n], dtype=np.float64).reshape(4, -1).T for n in names], axis=1)
    gram = S.T @ S
    sn = np.diag(gram).copy()
    cross = S.T @ Z
    active = sn > SILENT_TARGET_ENERGY
    ga = gram[np.ix_(active, active)]
    if ga.size and np.linalg.cond(ga) < 1e8:
        solved = np.linalg.solve(ga, cross[active])
        pn = np.sum(cross[active] * solved, axis=0)
        rank = int(active.sum())
    else:
        u, sv, _ = np.linalg.svd(S, full_matrices=False)
        keep = sv > sv[0] * 1e-10 if sv.size else np.zeros(0, dtype=bool)
        uz = u[:, keep].T @ Z
        pn = np.sum(uz * uz, axis=0)
        rank = int(keep.sum())
    zn = np.einsum('ij,ij->j', Z, Z)
    out = {}
    for q, name in enumerate(names):
        rows = []
        for i, stem in enumerate(SOURCES):
            j = q * 4 + i
            if sn[i] <= SILENT_TARGET_ENERGY:
                rows.append({'stem': stem, 'state': 'silent-target', 'target_gain': None,
                             'si_sdr': None, 'si_sir': None, 'si_sar': None, 'sd_sdr': None,
                             'output_rms': float(np.sqrt(zn[j] / len(S)))})
                continue
            gain = cross[i, j] / sn[i]
            tn = gain * gain * sn[i]
            floor = 1e-12 * max(sn[i], zn[j])
            values, states = {}, {}
            for metric, num, den in [('si_sdr', tn, zn[j]-tn), ('si_sir', tn, pn[j]-tn),
                                     ('si_sar', pn[j], zn[j]-pn[j]), ('sd_sdr', tn, zn[j]+sn[i]-2*cross[i,j])]:
                v = _db_ratio(float(max(num, 0)), float(max(den, 0)), floor=float(floor), db_cap=120.)
                values[metric], states[metric] = v['value_db'], v['state']
            rows.append({'stem': stem, 'state': 'ok', **values, 'metric_states': states, 'target_gain': float(gain),
                         'target_rms': float(np.sqrt(sn[i]/len(S))), 'output_rms': float(np.sqrt(zn[j]/len(S))), 'rank': rank})
        coefficients = np.divide(cross[:,q*4:(q+1)*4], sn[:,None], out=np.zeros((4,4)), where=sn[:,None]>SILENT_TARGET_ENERGY)
        out[name] = {'metrics': rows, 'projection': coefficients.tolist()}
    return out


def equivalence_check(targets, estimate):
    from experiments.banelp.metrics import project_metrics
    fast = score_sets(targets, {'output': estimate})['output']['metrics']
    errors = []
    for i, row in enumerate(fast):
        old = project_metrics(targets[i], estimate[i], targets)
        if old['state'] == 'silent-target':
            if row['state'] != 'silent-target':
                raise ValueError('silence-state mismatch')
            continue
        for key in ('si_sdr','si_sir','si_sar','sd_sdr'):
            a, b = row[key], old[key]['value_db']
            if (a is None) != (b is None):
                raise ValueError('metric state mismatch')
            if a is not None:
                errors.append(abs(a-b))
    if errors and max(errors) > 1e-8:
        raise ValueError(f'metric equivalence failed: {max(errors)}')
    return max(errors, default=0.)
