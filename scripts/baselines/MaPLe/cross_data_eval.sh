for SEED in 1 2 3
do
    # bash scripts/MaPLe/xd_test.sh BTMRI1 ${SEED}
    bash scripts/MaPLe/xd_test.sh CHMNIST ${SEED}
    bash scripts/MaPLe/xd_test.sh CCBTM ${SEED}
    bash scripts/MaPLe/xd_test.sh covid ${SEED}
    bash scripts/MaPLe/xd_test.sh ctkidney ${SEED}
    bash scripts/MaPLe/xd_test.sh dermamnist ${SEED}
    bash scripts/MaPLe/xd_test.sh lungcolon ${SEED}
    bash scripts/MaPLe/xd_test.sh retina ${SEED}
    bash scripts/MaPLe/xd_test.sh kneexray ${SEED}
    bash scripts/MaPLe/xd_test.sh octmnist ${SEED}
    bash scripts/MaPLe/xd_test.sh kvasir ${SEED}
    bash scripts/MaPLe/xd_test.sh CHMNISTv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh CCBTMv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh covidv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh ctkidneyv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh dermamnistv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh lungcolonv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh retinav2 ${SEED}
    bash scripts/MaPLe/xd_test.sh kneexrayv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh octmnistv2 ${SEED}
    bash scripts/MaPLe/xd_test.sh kvasirv2 ${SEED}
done