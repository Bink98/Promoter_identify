from Bio import SeqIO
import numpy as np
import os


os.chdir(r"E:\Users\Bink\Documents\iGEM\panda\Promoter_identify")
print(os.getcwd())

# Only run once
for record in SeqIO.parse("data/Ecoli_strain_DA33133.fasta", "fasta"):
    with open('data/Ecoli_strain_DA33133_trim500000_l61.fa', 'a') as newfile:
        i = 0
        for i in np.arange(500000):
            print(i)
            newfile.write(">" + str(record.id) + "   " + str(i) + "\n")
            newfile.write(str(record.seq)[i:61+i] + "\n")
