import os
import sys
import argparse
import sys
import tempfile

# Define file arguments
onto_file = str (sys.argv[1])
meta_file = tempfile.NamedTemporaryFile()
dict_file = str (sys.argv[2])

# Extract metadata
print ("\n1.Extract metadata\n")
cmd_meta= "groovy getMetadata.groovy " + str (onto_file)+ " all " + str(meta_file.name)
os.system(cmd_meta)

# Extract synonyms and labels 
print ("\n2.Build dictionary\n")
cmd_dict = "perl extr_dict.pl " + str (meta_file.name) + " > " +  dict_file
os.system(cmd_dict)

# Close files
meta_file.close()
