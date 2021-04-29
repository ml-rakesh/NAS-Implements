import os
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import GeneralNetwork
from utils import accuracy_metrics, f1, accuracy, f1_

logger = logging.getLogger('DARTS')

if __name__ == "__main__":
	parser = ArgumentParser("darts")
	parser.add_argument("--batch-size", default=128, type=int)
	parser.add_argument("--log-frequency", default=100, type=int)
	parser.add_argument("--visualization", default=False, action="store_true")
	parser.add_argument("--num_epochs", default=100, type=int)
	parser.add_argument("--train_mode", default='search', type=str)
	parser.add_argument("--arch_path", default='./final_architecture.json', type=str)
	parser.add_argument("--last_activation", default='softmax', type=str)
	parser.add_argument("--num_layers", default=6, type=int)
	parser.add_argument("--input_shape", default=0, type=int)
	parser.add_argument("--out_shape", default=0, type=int)
	parser.add_argument("--num_classes", default=2, type=int)
	parser.add_argument("--regression", default=False, type=bool)

	args = parser.parse_args()

	train_dataset, test_dataset = datasets.load_data(args.batch_size, args.train_mode)

	# model import
	model = GeneralNetwork(n_layers = args.num_layers,
							input_shape = args.input_shape,
							out_shape = args.out_shape,
							num_classes = args.num_classes,
							regression = args.regression,
							last_act = args.last_activation)

	# If classes are not eqully distributed then give them weights(each class)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	weight=(torch.tensor([0.5,4.0], dtype=torch.float)).to(device)
	criterion = nn.CrossEntropyLoss(weight=weight)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.001)

	if args.train_mode == 'search':
		from nni.algorithms.nas.pytorch.darts import DartsTrainer
		from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
		trainer = DartsTrainer(model,
								loss = criterion,
								metrics = lambda output, target: f1_(output, target),
								optimizer = optimizer,
								device   = device,
								batch_size = args.batch_size,
								num_epochs = args.num_epochs,
								dataset_train = train_dataset,
								dataset_valid = test_dataset,
								log_frequency = args.log_frequency)  
		if args.visualization:
			trainer.enable_visualization()
		logger.info('Start to train with DARTS...')
		trainer.train()
		logger.info('Training done')
		trainer.export(file=args.arch_path)
		logger.info('Best architecture exported in %s', args.arch_path)
        
	elif args.train_mode == 'retrain':
		from retrain import Retrain
		from nni.nas.pytorch.fixed import apply_fixed_architecture
		apply_fixed_architecture(model, args.arch_path)
		train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    pin_memory=True)
		test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=8,
                                                    pin_memory=True)
		trainer = Retrain(model, optimizer, criterion, lr_scheduler, device, train_loader, test_loader, n_epochs=args.num_epochs)
		trainer.run()
