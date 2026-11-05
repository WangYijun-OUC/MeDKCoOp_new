for SEED in 1 2 3
do
    # bash scripts/cocoop/xd_test.sh BTMRI1 ${SEED}
    bash scripts/cocoop/xd_test.sh CHMNIST ${SEED}
    bash scripts/cocoop/xd_test.sh CCBTM ${SEED}
    bash scripts/cocoop/xd_test.sh covid ${SEED}
    bash scripts/cocoop/xd_test.sh ctkidney ${SEED}
    bash scripts/cocoop/xd_test.sh dermamnist ${SEED}
    bash scripts/cocoop/xd_test.sh lungcolon ${SEED}
    bash scripts/cocoop/xd_test.sh retina ${SEED}
    bash scripts/cocoop/xd_test.sh kneexray ${SEED}
    bash scripts/cocoop/xd_test.sh octmnist ${SEED}
    bash scripts/cocoop/xd_test.sh kvasir ${SEED}
done