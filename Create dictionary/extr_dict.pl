#!/usr/bin/perl
#use strict;

#1. Define file argument
my $metadata_file = $ARGV[0];


#2. Build map of frequent words
my $freq_file ="frequent_lemma.num";
my @freq_words =();
open (FILE, $freq_file);
while (my $line1 =<FILE>)
{
	chomp ($line1);
	if ($line1 =~/^\S+\s+\S+\s+(\S+)/)
	{
		my $term = $1;
		push @freq_words, $term;
	}
}
my %freq_map = map {$_ => 1} @freq_words;

#3. Build mwt dictionary
print ("<?xml version='1.0' encoding='UTF-8'?>\n<mwt>s\n");
print ("<template><z:annotation fb=\"%1\" ids=\"%2\">%0</z:annotation></template>\n");
open (FH, $metadata_file);
while (my $line =<FH>)
{
	chomp ($line);	
	if ($line =~/<http:\/\/purl.obolibrary.org\/obo\/(\S+)>\s+(\S+)\s+(.+)/)
	{
		my $class_URI=$1;
		my $property=$2;
		# Process text
		my $literal =lc($3);
		$literal =~ s/[^a-zA-Z0-9,\s\t<>_:]//g;
		$literal =~s/[.]+//g;
		#Check if property is label or synonym
		if ($property=~/Synonym/ or $property =~/rdfs:label/)
		{	
			# Check if literal is a frequent word
			if (not(exists  $freq_map{$literal}))
			{
				# Create annotation		
				print ("<t p1=\"0\" p2=\"$class_URI\">$literal</t>\n");		
			}		
		}	
	}
	
}
print ("<template>%0</template>\n");
print ("<r><z:[^>]*>(.*</z)!:[^>]*></r>\n");
print ("<r>(([+\\-]|\\+/-|-/\\\\+)[\\r\\n\\t ]*)?[0-9]+([,.][0-9][0-9][0-9])*([.][0-9]*)?[\\r\\n\\t ]*(%|[\$]|([A-Za-z]+(/[A-Za-z]+)*))?</r>\n");
print ("<r>[Ff]ig(ures?|s?[.])[\\r\\n\\t ]*[0-9]+[A-Za-z]*</r>\n");
print ("<r>&lt;/?[A-Za-z_0-9\\-]+[^&gt;]+&gt;</r>\n");
print ("</mwt>");
