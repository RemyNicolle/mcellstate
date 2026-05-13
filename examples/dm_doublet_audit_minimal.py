import numpy as np
from scipy import sparse

from mcellstate_doublet import audit_doublets, make_prior


rng = np.random.default_rng(1)
profiles = np.array(
    [
        [0.70, 0.20, 0.05, 0.05],
        [0.05, 0.05, 0.20, 0.70],
        [0.38, 0.12, 0.12, 0.38],
    ]
)
labels = np.repeat([0, 1, 2], [20, 20, 8])
rows = [rng.multinomial(100, profiles[k]) for k in labels]
X = sparse.csr_matrix(np.asarray(rows))

psi = make_prior(X, tau=1.0)
res = audit_doublets(X, labels, psi=psi, top_m_parents=2, doublet_rate=0.05)
print(res.cluster_table)
