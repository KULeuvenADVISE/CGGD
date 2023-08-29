CC = python3.8
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all plot train pack view metrics report

K = 2
BS = 4

G_RGX = (\d+_Case\d+_\d+)_\d+
NET = ResidualUNet
SAVE = --save_train
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

NETWORK_SEED1 = 50
NETWORK_SEED2 = 60
NETWORK_SEED3 = 70
NETWORK_SEED4 = 80

GPU_NUMBER = 0

TRN = results/prostate_10/cggd_box_prior_box_size \
    results/prostate_10/penalty_box_prior_box_size \
    results/prostate_10/logbarrier_box_prior_box_size \
    results/prostate_10/fs

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives

train: $(TRN)

# Weak labels generation
weaks = data/PROSTATE_10/train/centroid data/PROSTATE_10/val/centroid \
		data/PROSTATE_10/train/erosion data/PROSTATE_10/val/erosion \
		data/PROSTATE_10/train/random data/PROSTATE_10/val/random \
		data/PROSTATE_10/train/box data/PROSTATE_10/val/box \
		data/PROSTATE_10/train/thickbox data/PROSTATE_10/val/thickbox

weak: $(weaks)

data/PROSTATE_10/train/centroid data/PROSTATE_10/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/PROSTATE_10/train/erosion data/PROSTATE_10/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/PROSTATE_10/train/random data/PROSTATE_10/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/PROSTATE_10/train/box data/PROSTATE_10/val/box: OPT = --seed=0 --margin=0 --strategy=box_strat --allow_bigger --allow_overflow
data/PROSTATE_10/train/thickbox data/PROSTATE_10/val/thickbox: OPT = --seed=0 --margin=10 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): data/PROSTATE_10
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@

data/PROSTATE_10-Aug/train/box data/PROSTATE_10-Aug/val/box: | data/PROSTATE_10-Aug
data/PROSTATE_10-Aug/train/gt data/PROSTATE_10-Aug/val/gt: data/PROSTATE_10-Aug
data/PROSTATE_10-Aug: data/PROSTATE_10 | weak
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@


results/prostate_10/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1), \
    ('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 0), \
    ('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 0)]" \
    --box_prior --box_prior_args "{'idc': [1], 'd': 5}"
results/prostate_10/fs: data/PROSTATE_10-Aug/train/gt data/PROSTATE_10-Aug/val/gt
results/prostate_10/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate_10/fs: EPC= --n_epoch=100

results/prostate_10/cggd_box_prior_box_size: OPT = --losses="[('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorLogBarrier', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate_10/cggd_box_prior_box_size: data/PROSTATE_10-Aug/train/box data/PROSTATE_10-Aug/val/box
results/prostate_10/cggd_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate_10/cggd_box_prior_box_size: CGGD = --cggd
results/prostate_10/cggd_box_prior_box_size: CGGD_BOX_PRIOR = --cggd_box_prior
results/prostate_10/cggd_box_prior_box_size: CGGD_BOX_SIZE = --cggd_box_size
results/prostate_10/cggd_box_prior_box_size: L_RATE = --l_rate=5e-3
results/prostate_10/cggd_box_prior_box_size: EPC = --n_epoch=500

results/prostate_10/logbarrier_box_prior_box_size: OPT = --losses="[('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorLogBarrier', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate_10/logbarrier_box_prior_box_size: data/PROSTATE_10-Aug/train/box data/PROSTATE_10-Aug/val/box
results/prostate_10/logbarrier_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate_10/logbarrier_box_prior_box_size: EPC = --n_epoch=100

results/prostate_10/penalty_box_prior_box_size: OPT = --losses="[('BoxPriorNaivePenalty', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('NaivePenalty', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorNaivePenalty', 'NaivePenalty'], 'mu': 1.1}" --temperature 1
results/prostate_10/penalty_box_prior_box_size: data/PROSTATE_10-Aug/train/box data/PROSTATE_10-Aug/val/box
results/prostate_10/penalty_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate_10/penalty_box_prior_box_size: EPC = --n_epoch=100

results/prostate_10/%:
	rm -rf $@_tmp_$(NETWORK_SEED1)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--in_memory --compute_3d_dice $(SAVE) \
		--workdir=$@_tmp_$(NETWORK_SEED1) --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED1) $(EPC) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(CGGD_BOX_PRIOR) $(CGGD_BOX_SIZE) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED1) $@_$(NETWORK_SEED1)
	rm -rf $@_tmp_$(NETWORK_SEED2)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--in_memory --compute_3d_dice $(SAVE) \
		--workdir=$@_tmp_$(NETWORK_SEED2) --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED2) $(EPC) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(CGGD_BOX_PRIOR) $(CGGD_BOX_SIZE) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED2) $@_$(NETWORK_SEED2)
	rm -rf $@_tmp_$(NETWORK_SEED3)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--in_memory --compute_3d_dice $(SAVE) \
		--workdir=$@_tmp_$(NETWORK_SEED3) --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED3) $(EPC) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(CGGD_BOX_PRIOR) $(CGGD_BOX_SIZE) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED3) $@_$(NETWORK_SEED3)
	rm -rf $@_tmp_$(NETWORK_SEED4)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--in_memory --compute_3d_dice $(SAVE) \
		--workdir=$@_tmp_$(NETWORK_SEED4) --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED4) $(EPC) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(CGGD_BOX_PRIOR) $(CGGD_BOX_SIZE) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED4) $@_$(NETWORK_SEED4)

