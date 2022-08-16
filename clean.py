import pandas as pd
import re
import html
#%%
am = pd.read_csv('amendements.csv',error_bad_lines=False, sep=";")
am.set_index(['id'], inplace = True)
am = am[['sort','groupe','expose','corps','objet','avis_gouv','reponse_complete', 'article']]

#%%
def clean_html_amendements(am1):
    am1['expose'] = am1['expose'].apply(lambda x: html_to_string(x) if str(x) != 'nan' else x)
    am1['objet'] = am1['objet'].apply(lambda x: html_to_string(x) if str(x) != 'nan' else x)
    am1['reponse_complete'] = am1['reponse_complete'].apply(lambda x: html_to_string(x) if str(x) != 'nan' else x)

    am1 = am1.dropna()
    am1 = am1.drop_duplicates()
    am1['corps_bis'] = am1['corps'].apply(lambda x: table(x) if str(x) != 'nan'  else x)
    am1['corps_clean'] = am1['corps_bis'].apply(lambda x: ' '.join([value  if type(value) == str else "TABLE"+str(key) for key,value in x.items() ]))
    am1['corps_table'] = am1['corps_bis'].apply(lambda x: {key:value for key,value in x.items() if type(value) != str })
    del am1['corps']
    del am1['corps_bis']
    return am1

def table(texte):
    #texte = unicodedata.normalize("NFKD",texte)
    dico = {}
    li1 = texte.replace("<table","*<table").replace("/table>","/table>*").split("*")
    i = 0
    for x in li1 :
        if "<table" in x:
            try:
                dico[i] = pd.read_html(x)[0]
            except:
                pass
        else:
            dico[i] = html_to_string(x)
        i+=1
    return dico

def html_to_string(text):
    return html.unescape(re.sub('<[^<]+?>', '', text))

#%%
am = clean_html_amendements(am)



