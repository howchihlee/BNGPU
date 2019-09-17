mkdir -p GNWeaver
wget https://genome.nyumc.org/userdata/mas02/supplemental_ODLP/data/GNW_YEAST.zip --no-check-certificate
zip -FF GNW_YEAST.zip  --out GNW_YEAST_repaired.zip 
unzip GNW_YEAST_repaired.zip GNW_YEAST/data_obs.mat GNW_YEAST/gene_names.txt GNW_YEAST/GS.mat
mv GNW_YEAST ./GNWeaver
rm GNW_YEAST.zip GNW_YEAST_repaired.zip 
