# Ontology-based-normalization
We propose a workflow that can be used to normalize natural language text and ontology meta-data with ontology classes. The normalized texts can then be used to learn feature representations of ontology classes and biomedical entities.
The normalization workflow takes a dictionary mapping "a set of words" to an ontology ID and runs a normalization pipeline based on Whatizit https://www.ebi.ac.uk/webservices/whatizit/info.jsf. 
An updated version of our representation learning model, OPA2Vec, is then applied on the normalized text to learn feature vectors of classes and entities.
A deep siamese neural network can then be applied on the feature vectors to learn unknown associations between biomedical entities.

Below is a description of the code and data included in this repository. 
## Dictionary creation 
To create a dictionary based on an ontology *Onto.owl*, under the folder "Create dictionary" run from the terminal:
```
python create_dictionary.py Onto.owl dict.mwt
```
where *Onto.owl* is the ontology in OWL formal and *dict.mwt* is the output file where you want to save the dictionary.
## Text annotation 
Save your dictionary to *FullTextAnnotation/automata*
In *FullTextAnnotation/scripts/FT_Annotator.sh*, modify 
## Representation learning
## Prediction
## Data 
