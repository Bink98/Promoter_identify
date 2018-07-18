from Bio import SeqIO
import numpy as np
import os


os.chdir(r"E:/Users/Bink/Documents/iGEM/panda")
print(os.getcwd())

# Only run once
for record in SeqIO.parse("data/escherichia coli.fasta", "fasta"):
    with open('data/Ecoli_genome_trim.fa', 'a') as newfile:
        i = 0
        for i in np.arange(500):
            newfile.write(">" + str(record.id) + "   " + str(i) + "\n")
            newfile.write(str(record.seq)[i:81+i] + "\n")