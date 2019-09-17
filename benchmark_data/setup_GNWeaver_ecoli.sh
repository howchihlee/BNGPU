mkdir -p GNWeaver
wget https://genome.nyumc.org/userdata/mas02/supplemental_ODLP/data/GNW_ECOLI.zip --no-check-certificate
unzip GNW_ECOLI.zip GNW_ECOLI/data_obs.mat GNW_ECOLI/gene_names.txt GNW_ECOLI/GS.mat
mv GNW_ECOLI ./GNWeaver
rm GNW_ECOLI.zip