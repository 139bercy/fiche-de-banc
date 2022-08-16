from sentence_transformers import SentenceTransformer

#%% on récupère l'article concerné 
am['article_clean'] = am['article'].apply(lambda x : ' '.join([ele for ele in x.split() if ele.isdigit() or ele == 'liminaire' or ele == 'Ier']))

#%% modèle d'embedding
model = SentenceTransformer("dangvantuan/sentence-camembert-large")

#%% distance cosinus
import numpy as np
from numpy.linalg import norm
def cos(A,B):
    return  np.dot(A,B)/(norm(A)*norm(B))

#%%
def article_commun(amendements, amendement):
    return amendements[amendements['article_clean'] == list(amendement['article_clean'])[0]]


#%% on choisit un amendement au hasard
amendement = am.sample()
am.drop(am[am.index == amendement.index[0] ].index, inplace=True)

#%%
voisins = article_commun(am, amendement)
vecteur_amendement = model.encode(list(amendement['objet'])[0])
voisins['score'] = voisins['objet'].apply(lambda x: cos(model.encode(x),vecteur_amendement))

