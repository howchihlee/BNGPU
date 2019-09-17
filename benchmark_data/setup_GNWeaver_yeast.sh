mkdir -p GNWeaver
wget https://genome.nyumc.org/userdata/mas02/supplemental_ODLP/data/GNW_YEAST.zip --no-check-certificate
unzip GNW_YEAST.zip GNW_YEAST/data_obs.mat GNW_YEAST/gene_names.txt GNW_YEAST/GS.mat
mv GNW_ECOLI ./GNWeaver
rm GNW_YEAST.zip