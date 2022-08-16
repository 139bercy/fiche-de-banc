from haystack.nodes import FARMReader
from unidecode import unidecode
#%% Modèle initial
reader = FARMReader(model_name_or_path="etalab-ia/camembert-base-squadFR-fquad-piaf", use_gpu=True)

#%% Modèle fine-tuné
#reader.train(data_dir="/home/pierricspery/Bureau/fdb", train_filename="test.json", use_gpu=True, n_epochs=1, save_dir="my_model")
new_reader = FARMReader(model_name_or_path="my_model")

#%% Comparaison d'objets
print(unidecode(new_reader.predict_on_texts(texts=[am['expose'][i]],question =["comment convient-il de faire ?","que propose ou vise cet amendement ?"])['answers'][0].to_dict()['answer']))
print(unidecode(reader.predict_on_texts(texts=[am['expose'][i]],question =["comment convient-il de faire ?","que propose ou vise cet amendement ?"])['answers'][0].to_dict()['answer']))
