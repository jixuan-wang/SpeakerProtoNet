import random

spk2utt = {}
lines = open('/h/jixuan/gobi1/asr/espnet/tools/kaldi/egs/voxceleb/v2/data/voxceleb2_test/spk2utt', 'r').readlines()

trials_out = open('/h/jixuan/gobi1/asr/espnet/tools/kaldi/egs/voxceleb/v2/data/voxceleb2_test/trials', 'w')

for line in lines:
    temp = line.strip().split()
    sid = temp[0]
    utt = temp[1:]
    spk2utt[sid] = utt

for line in lines:
    temp = line.strip().split()
    sid = temp[0]
    utt = temp[1:]

    for u in utt:
        for i in range(8):
            if i % 2 == 0:
                # target
                while True:
                    rand_u = random.choice(spk2utt[sid])
                    if u != rand_u:
                        break
                trials_out.write(f"{u} {rand_u} target\n")
            else:
                # nontarget
                while True:
                    rand_s = random.choice(list(spk2utt.keys()))
                    if sid != rand_s:
                        break
                trials_out.write(f"{u} {random.choice(spk2utt[rand_s])} nontarget\n")
trials_out.close()






