CC = python3.8
SHELL = zsh
PP = PYTHONPATH="$(PYTHONPATH):."

EPC = 50
NET = ENet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NETWORK_SEED1 = 50
NETWORK_SEED2 = 60
NETWORK_SEED3 = 70
NETWORK_SEED4 = 80

GPU_NUMBER = 0

TRN = results/toy_50/fs \
	results/toy_50/penalty_size_centroid \
	results/toy_50/logbarrier_size_centroid \
	results/toy_50/cggd_size_centroid

GRAPH = results/toy_50/val_dice.png results/toy_50/tra_dice.png \
	results/toy_50/val_loss.png results/toy_50/tra_loss.png
HIST =  results/toy_50/val_dice_hist.png
BOXPLOT = results/toy_50/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives

plot: $(PLT)

data/TOY_50: data/TOY_50/train/gt data/TOY_50/val/gt
DTS = data/TOY_50
data/TOY_50/train/gt data/TOY_50/val/gt:
	rm -rf $(DTS)_tmp $(DTS)
	$(PP) $(CC) $(CFLAGS) preprocess/gen_toy.py --dest $(DTS)_tmp -n 1000 100 -r 25 -wh 256 256
	mv $(DTS)_tmp $(DTS)

data/TOY_50/train/gt data/TOY_50/val/gt:

# Trainings
results/toy_50/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1), \
    ('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 0), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 0)]"
results/toy_50/fs: data/TOY_50/train/gt data/TOY_50/val/gt
results/toy_50/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True), ('gt', gt_transform, True)]"

## No labeled pixels
results/toy_50/penalty_size_centroid: OPT = --losses="[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy_50/penalty_size_centroid: data/TOY_50/train/gt data/TOY_50/val/gt
results/toy_50/penalty_size_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"

results/toy_50/logbarrier_size_centroid: OPT = --losses="[('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 5}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy_50/logbarrier_size_centroid: data/TOY_50/train/gt data/TOY_50/val/gt
results/toy_50/logbarrier_size_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"

results/toy_50/cggd_size_centroid: OPT = --losses="[('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2)]"
results/toy_50/cggd_size_centroid: data/TOY_50/train/gt data/TOY_50/val/gt
results/toy_50/cggd_size_centroid: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True)]"
results/toy_50/cggd_size_centroid: CGGD = --cggd
results/toy_50/cggd_size_centroid: SIZE_CON = --size_con
results/toy_50/cggd_size_centroid: CENTROID_CON = --centroid_con
results/toy_50/cggd_size_centroid: L_RATE = --l_rate=5e-3

# Template
results/toy_50/%:
	rm -rf $@_tmp_$(NETWORK_SEED1)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --schedule --save_train --temperature 5 \
		--n_epoch=$(EPC) --workdir=$@_tmp_$(NETWORK_SEED1) --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED1) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(SIZE_CON) $(CENTROID_CON) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED1) $@_$(NETWORK_SEED1)
	rm -rf $@_tmp_$(NETWORK_SEED2)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --schedule --save_train --temperature 5 \
		--n_epoch=$(EPC) --workdir=$@_tmp_$(NETWORK_SEED2) --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED2) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(SIZE_CON) $(CENTROID_CON) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED2) $@_$(NETWORK_SEED2)
	rm -rf $@_tmp_$(NETWORK_SEED3)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --schedule --save_train --temperature 5 \
		--n_epoch=$(EPC) --workdir=$@_tmp_$(NETWORK_SEED3) --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED3) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(SIZE_CON) $(CENTROID_CON) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED3) $@_$(NETWORK_SEED3)
	rm -rf $@_tmp_$(NETWORK_SEED4)
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --schedule --save_train --temperature 5 \
		--n_epoch=$(EPC) --workdir=$@_tmp_$(NETWORK_SEED4) --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) --network_seed=$(NETWORK_SEED4) $(OPT) $(DATA) $(DEBUG) $(CGGD) $(SIZE_CON) $(CENTROID_CON) $(L_RATE) --gpu_number=$(GPU_NUMBER)
	mv $@_tmp_$(NETWORK_SEED4) $@_$(NETWORK_SEED4)

