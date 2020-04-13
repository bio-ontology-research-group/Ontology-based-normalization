# Ontology-based-normalization
We propose a workflow that can be used to normalize natural language text and ontology meta-data with ontology classes. The normalized texts can then be used to learn feature representations of ontology classes and biomedical entities.
The normalization workflow takes a dictionary mapping "a set of words" to an ontology ID and runs a normalization pipeline based on Whatizit https://www.ebi.ac.uk/webservices/whatizit/info.jsf. 
An updated version of our representation learning model, OPA2Vec, is then applied on the normalized text to learn feature vectors of classes and entities.
A deep siamese neural network can then be applied on the feature vectors to learn unknown associations between biomedical entities.

Below is a description of the code and data included in this repository. 
## Dictionary creation 
To create a dictionary based on an ontology *Onto.owl*, under the folder *Create dictionary* run from the terminal:
```
python create_dictionary.py Onto.owl dict.mwt
```
where *Onto.owl* is the ontology in OWL format and *dict.mwt* is the output file where you want to save the dictionary (in mwt format).
## Text annotation 
- Download the code from https://bio2vec.cbrc.kaust.edu.sa/data/Full-Text-Annotation/
- Save your dictionary to *FullTextAnnotation/automata*
- In *FullTextAnnotation/scripts/FT_Annotator.sh*, modify the line :
*-cp $OTHERS monq.programs.DictFilter -t elem -e plain -ie UTF-8 -oe UTF-8 $DICXX/Swissprot_cttv_Oct2016.mwt"*
by replacing *Swissprot_cttv_Oct2016.mwt* withe dictionary of your choice in mwt format.
- Run the following command:
```
zcat Path_2_Corpora/Some_FullText_File.xml.gz|sh FullTextAnnotation/scripts/FT_Annotator.sh |gzip >OutputFile.xml.gz
```
where *Some_FullText_File.xml.gz* is the text corpus you want to normalize (annotate)
## Representation learning
The updated representation learning algorithm is available in *UPDATED_OPA2VEC/*. For annotation add *-annotate_metadata yes* to the terminal command. 
## Prediction
The deep neural networks used for prediction are available in *Siamese_NeuralNetworks/*
## Data 
The dictionaries used and the prediction results are available in the *data/* folder
