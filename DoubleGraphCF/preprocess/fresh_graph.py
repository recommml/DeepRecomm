import numpy as np
import scipy.sparse as sp

ug = sp.load_npz('Data/amazon-book/user_graph.npz.bk')
ig = sp.load_npz('Data/amazon-book/item_graph.npz.bk')
lug, lig = ug.shape[0], ig.shape[0]
ug = ug.tocsr()
ig = ig.tocsr()
ux, uy, uz = sp.find(ug)
ix, iy, iz = sp.find(ig)
nug = sp.dok_matrix((lug, lug), dtype=np.float32)
nig = sp.dok_matrix((lig, lig), dtype=np.float32)
nug = nug.tolil()
nig = nig.tolil()
ug = ug.tolil()
ig = ig.tolil()

uids = np.where(uz >= 10)[0]
for uid in uids:
    tux, tuy = ux[uid], uy[uid]
    nug[tux, tuy] = ug[tux, tuy]

iids = np.where(iz >= 10)[0]
for iid in iids:
    tix, tiy = ix[iid], iy[iid]
    nig[tix, tiy] = ig[tix, tiy]

nug = nug.tocsr()
nig = nig.tocsr()
sp.save_npz('Data/amazon-book/user_graph.npz', nug)
sp.save_npz('Data/amazon-book/item_graph.npz', nig)
