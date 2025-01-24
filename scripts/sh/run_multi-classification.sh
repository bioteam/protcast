#!/bin/bash
# Run MultiClassifier on multiple sequence sets
#SBATCH --job-name run_multi_classification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_multi_classification-rtx.out
#SBATCH -e run_multi_classification-rtx.err
#SBATCH -n 20

DIR=multi-classification
NAME=rtx

source ${HOME}/.bash_profile
source ${HOME}/tf282/bin/activate

cd ${HOME}/git/ProtCast/

ALGORITHMS=(aac apaac cksaagp cksaap ctdc ctdd ctdt ctriad dde dpc gaac gdpc geary gtpc ksctriad moran nmbroto paac qsorder socnumber tpc)
#ALGORITHMS=(tpc socnumber qsorder paac nmbroto moran ksctriad gtpc geary gdpc gaac dpc dde ctriad ctdt ctdd ctdc cksaap cksaagp apaac aac)
#ALGORITHMS=(ctriad ctdt ctdd)

for ALG in "${ALGORITHMS[@]}"; do
      if [ -s $DIR/${ALG}.tsv ]; then
         continue
      fi
      time python3 scripts/make_multi_class_model.py -s test/data/random-level-4.fa -a $ALG -f ifeatpro
      time python3 scripts/run_multi_class_inference.py -s test/data/random-level-4.fa -g *GOEncoder.pickle -m *${ALG}.h5 | grep -v 'ms/step' | grep -v Descriptor | grep -v Error > ${ALG}_${NAME}.tsv 
      mv ${ALG}_${NAME}.tsv ${DIR}
      rm -f *h5 *pickle
   done
done
