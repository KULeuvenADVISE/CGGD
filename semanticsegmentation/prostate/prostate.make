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

TRN = results/prostate/cggd_box_prior_box_size \
    results/prostate/penalty_box_prior_box_size \
    results/prostate/logbarrier_box_prior_box_size \
    results/prostate/fs

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz

all: pack

plot: $(PLT)

train: $(TRN)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/PROSTATE/train/gt data/PROSTATE/val/gt: data/PROSTATE
data/PROSTATE: data/promise
	rm -rf $@_tmp
	$(PP) $(CC) $(CFLAGS) preprocess/slice_promise.py --source_dir $< --dest_dir $@_tmp --n_augment=0
	mv $@_tmp $@
data/promise: data/prostate_v2.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@


# Weak labels generation
weaks = data/PROSTATE/train/centroid data/PROSTATE/val/centroid \
		data/PROSTATE/train/erosion data/PROSTATE/val/erosion \
		data/PROSTATE/train/random data/PROSTATE/val/random \
		data/PROSTATE/train/box data/PROSTATE/val/box \
		data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox

weak: $(weaks)

data/PROSTATE/train/centroid data/PROSTATE/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/PROSTATE/train/erosion data/PROSTATE/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/PROSTATE/train/random data/PROSTATE/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/PROSTATE/train/box data/PROSTATE/val/box: OPT = --seed=0 --margin=0 --strategy=box_strat --allow_bigger --allow_overflow
data/PROSTATE/train/thickbox data/PROSTATE/val/thickbox: OPT = --seed=0 --margin=10 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): data/PROSTATE
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@


data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box: | data/PROSTATE-Aug
data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt: data/PROSTATE-Aug
data/PROSTATE-Aug: data/PROSTATE | weak
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@


results/prostate/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1), \
    ('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 0), \
    ('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 0)]" \
    --box_prior --box_prior_args "{'idc': [1], 'd': 5}"
results/prostate/fs: data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt
results/prostate/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate/fs: EPC = --n_epoch=100

results/prostate/cggd_box_prior_box_size: OPT = --losses="[('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorLogBarrier', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/cggd_box_prior_box_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/cggd_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate/cggd_box_prior_box_size: CGGD = --cggd
results/prostate/cggd_box_prior_box_size: CGGD_BOX_PRIOR = --cggd_box_prior
results/prostate/cggd_box_prior_box_size: CGGD_BOX_SIZE = --cggd_box_size
results/prostate/cggd_box_prior_box_size: EPC = --n_epoch=150

results/prostate/logbarrier_box_prior_box_size: OPT = --losses="[('BoxPriorLogBarrier', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorLogBarrier', 'LogBarrierLoss'], 'mu': 1.1}" --temperature 1
results/prostate/logbarrier_box_prior_box_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/logbarrier_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate/logbarrier_box_prior_box_size: EPC = --n_epoch=100

results/prostate/penalty_box_prior_box_size: OPT = --losses="[('BoxPriorNaivePenalty', {'idc': [1], 't': 5}, None, None, None, 1e-2), \
	('NaivePenalty', {'idc': [1], 't': 5}, 'BoxBounds', {'margins': [0.50, 0.90]}, 'soft_size', 1e-2)]" \
	--box_prior --box_prior_args "{'idc': [1], 'd': 5}" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': ['BoxPriorNaivePenalty', 'NaivePenalty'], 'mu': 1.1}" --temperature 1
results/prostate/penalty_box_prior_box_size: data/PROSTATE-Aug/train/box data/PROSTATE-Aug/val/box
results/prostate/penalty_box_prior_box_size: DATA = --folders="$(B_DATA)+[('box', gt_transform, True), ('box', gt_transform, True)]"
results/prostate/penalty_box_prior_box_size: EPC = --n_epoch=100


results/prostate/%:
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
