#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

python test.py
#which nnet3-copy-egs
nnet3-copy-egs -h

#nnet3-copy-egs --frame=0  scp:/scratch/gobi1/jixuan/asr/espnet/tools/kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a/egs/egs.1.scp ark:- |  nnet3-shuffle-egs --buffer-size=1000 --srand=123 ark:- ark:- | nnet3-merge-egs --minibatch-size=32 ark:- ark,t: | cat > temp.txt
#nnet3-copy-egs --frame=0  scp:/scratch/gobi1/jixuan/asr/espnet/tools/kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a/egs/egs.1.scp ark,t: | head -n 10
