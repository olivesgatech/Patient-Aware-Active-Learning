import os
import pandas as pd
import torch
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from myUtils.models import MLP
from tqdm import tqdm
from Data.datasets import make_data_loader
from myUtils.saver import Saver, plot_batch
from myUtils.summaries import TensorboardSummary
from modeling.resnet import cResNet
from modeling.models import DenseNet, VGG, MLPclassifier
from myUtils.loss import PositiveCongruentLoss
from sklearn.metrics import precision_score, recall_score


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # load evaluation test forgetting events
        if (args.ef_eval_path is not None) and (args.switch_eval_path is not None):
            self.test_weights = np.load(self.args.ef_eval_path)
            self.test_switch_weights = np.load(self.args.switch_eval_path)
            self.weight_eval = True
        else:
            self.weight_eval = False

        # Define Dataloader
        # dataloader threads
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.kwargs = kwargs

        data_configs = make_data_loader(args, **kwargs)

        self.train_loader = data_configs['train_loader']
        self.test_loader = data_configs['test_loader']
        self.grad_loader = data_configs['grad_loader']
        self.unlabeled_loader = data_configs['unlabeled_loader']
        self.corr_loader = data_configs['corrupted_loader']
        self.nclasses = data_configs['nclasses']
        self.train_pool = data_configs['train_pool']
        self.test_pool = data_configs['test_pool']
        self.dim = data_configs['dim']

        # init arrays for statistics capture
        # learned array keeps track if sample was learned or seen at all
        self.learned = np.zeros(self.train_pool, dtype=int)
        self.unlabeled_seen = np.zeros(self.train_pool, dtype=int)

        # learned array keeps track if sample was learned at all
        self.test_learned = np.zeros(self.test_pool, dtype=int)

        # array for forgetting events and unlabeled switching stats
        self.forgetting_events = np.zeros(self.train_pool, dtype=int)
        self.unlabeled_variety = np.zeros(self.train_pool, dtype=int)

        # array for test forgetting events
        self.test_forgetting_events = np.zeros(self.test_pool, dtype=int)

        # accuracy and prediciton of previous epoch
        self.prev_acc = np.zeros(self.train_pool, dtype=int)
        self.unlabeled_prev_pred = np.zeros(self.train_pool, dtype=int)

        # accuracy of previous epoch test
        self.test_prev_acc = np.zeros(self.test_pool, dtype=int)

        # define if pretrained
        self.pretrained = self.args.pretrained

        # Setup model
        model = self.get_model(args.architecture)
        att_model = MLP(20)

        # define optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                        weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 110], gamma=0.2)
        else:
            print('Specified optimizer not recognized. Options are: adam and sgd')
        att_optimizer = torch.optim.Adam(att_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0, amsgrad=True)

        # Define Loss
        # if self.args.train_type == 'positive_congruent':
        #     self.criterion = PositiveCongruentLoss
        # else:
        if args.strategy == 'badge':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.att_criterion = nn.MSELoss()

        # init model and optimizer
        self.model, self.optimizer = model, optimizer
        self.att_model, self.att_optimizer = att_model, att_optimizer

        # Using cuda
        if args.cuda:
            # use multiple GPUs if available
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.att_model = torch.nn.DataParallel(self.att_model, device_ids=self.args.gpu_ids)
            # use all GPUs
            self.model = self.model.cuda()
            self.att_model = self.att_model.cuda()
            if self.args.train_type == 'traditional':
                self.criterion = self.criterion.cuda()
                self.att_criterion = self.att_criterion.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            # we have a checkpoint
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            # load checkpoint
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # minor difference if working with cuda
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def get_model(self, architecture):
        if architecture == 'resnet_18':
            model = cResNet(type=18, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'resnet_34':
            model = cResNet(type=34, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'resnet_50':
            model = cResNet(type=50, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'resnet_101':
            model = cResNet(type=101, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'resnet_152':
            model = cResNet(type=152, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'densenet_121':
            model = DenseNet(type=121, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'densenet_161':
            model = DenseNet(type=161, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'densenet_169':
            model = DenseNet(type=169, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'densenet_201':
            model = DenseNet(type=201, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'vgg_11':
            model = VGG(type=11, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'vgg_13':
            model = VGG(type=13, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'vgg_16':
            model = VGG(type=16, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'vgg_19':
            model = VGG(type=19, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'mlp':
            model = MLPclassifier(dim=self.dim, num_classes=self.nclasses)
        else:
            raise NotImplementedError

        return model

    def update_loaders(self, current_idxs):
        '''adds new samples to the current training pool.
        Parameters:
            :param new_samples: queried samples to be added to the training set
            :type current_idxs: ndarray'''
        # reinitialize loaders with the new training pool
        data_configs = make_data_loader(self.args, current_idxs=current_idxs, **self.kwargs)

        self.train_loader = data_configs['train_loader']
        self.test_loader = data_configs['test_loader']
        self.unlabeled_loader = data_configs['unlabeled_loader']
        self.corr_loader = data_configs['corrupted_loader']
        self.grad_loader = data_configs['grad_loader']
        self.nclasses = data_configs['nclasses']
        self.train_pool = data_configs['train_pool']
        self.test_pool = data_configs['test_pool']

    def clear_statistics(self):
        '''Clears all statistics captured in a previous round.'''
        # learned array keeps track if sample was learned at all
        self.learned = np.zeros(self.train_pool, dtype=int)

        # array for forgetting events
        self.forgetting_events = np.zeros(self.train_pool, dtype=int)

        # accuracy of previous epoch
        self.prev_acc = np.zeros(self.train_pool, dtype=int)

        # unlabeled statistics
        self.unlabeled_prev_pred = np.zeros(self.train_pool, dtype=int)
        self.unlabeled_variety = np.zeros(self.train_pool, dtype=int)
        self.unlabeled_seen = np.zeros(self.train_pool, dtype=int)

        # test statistics
        self.test_learned = np.zeros(self.test_pool, dtype=int)
        self.test_forgetting_events = np.zeros(self.test_pool, dtype=int)
        self.test_prev_acc = np.zeros(self.test_pool, dtype=int)

    def reset_model(self):
        '''Erases the current model for further training from scratch'''

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        if self.args.cuda:
            self.model = self.model.apply(weight_reset).cuda()
        else:
            self.model = self.model.apply(weight_reset)

    def training(self, epoch, save_checkpoint=False, output_dir=None, round=None):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self.model.train()
        # initializes cool bar for visualization
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        preds_and_output = np.array([], dtype=np.int64).reshape(0, 4)
        input_images = torch.tensor([]).cuda()
        old_outputs = torch.tensor([]).cuda()
        prev_preds = torch.tensor([]).cuda()
        # iterate over all samples in each batch i
        for i, (image, target, idxs) in enumerate(tbar):
            # convert target to one hot vectors
            one_hot = torch.zeros(target.shape[0], self.nclasses)
            one_hot[range(target.shape[0]), target.long()] = 1
            # assign each image and target to GPU
            if self.args.cuda:
                image, target = image.to(DEVICE), target.to(DEVICE)
                one_hot = one_hot.cuda()

            # plot_batch(image, DEVICE)

            # update model
            self.optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self.model(image)

            logit, pred = torch.max(output.data, 1)
            total += target.size(0)

            input_images = torch.cat((input_images, image), 0)
            old_outputs = torch.cat((old_outputs, output), 0)
            prev_preds = torch.cat((prev_preds, pred), 0)

            t1 = np.expand_dims(idxs.cpu().numpy(), axis=1)
            # t2 = np.expand_dims(id.cpu().numpy(), axis=1)
            t3 = np.expand_dims(pred.cpu().numpy(), axis=1)
            t4 = np.expand_dims(target.cpu().numpy(), axis=1)
            t5 = np.expand_dims(logit.cpu().numpy(), axis=1)
            collect_preds_and_output = np.concatenate([t1, t3, t4, t5], axis=1)
            preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])

            # collect forgetting events
            acc = pred.eq(target.data)
            delta = np.clip(self.prev_acc[idxs] - acc.cpu().numpy(), a_min=0, a_max=1)
            self.forgetting_events[idxs] += delta

            # mark learned samples
            self.learned[idxs] += acc.cpu().numpy()

            # update previous accuracy
            self.prev_acc[idxs] = acc.cpu().numpy()

            # Perform model update
            # calculate loss
            if self.args.train_type == 'positive_congruent' and round > 0:
                self.criterion = PositiveCongruentLoss
                OldOutputPath = os.path.join(self.args.base_dir, self.args.dataset + self.args.eval,
                                             self.args.architecture, self.args.start_strategy, self.args.out_dir,
                                             "pytorch_ckpt_" + str(round - 1) + ".tar")
                state_dict = torch.load(OldOutputPath)
                old_images = state_dict['input_images']
                old_logits = state_dict['old_output']
                old_preds = state_dict['old_preds']
                old_logits = torch.split(old_logits, self.args.batch_size)
                old_images = torch.split(old_images, self.args.batch_size)
                old_preds = torch.split(old_preds, self.args.batch_size)

                if i < len(old_images):
                    # print('i is: ' + str(i))
                    # print('hello')
                    loss = self.criterion(output, target, pred, self.model, old_logits[i], old_images[i], old_preds[i])
                else:
                    # print('byeee')
                    CE_loss = nn.CrossEntropyLoss()
                    loss = CE_loss(output, target)
            else:
                if self.args.strategy == 'badge':
                    loss = self.criterion(output, target)
                else:
                    loss = self.criterion(output, one_hot)
            # perform backpropagation
            loss.backward()

            # update params with gradients
            self.optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += pred.eq(target.data).cpu().sum()


        # Update optimizer step
        if self.args.optimizer == 'sgd':
            self.scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        # save checkpoint
        if save_checkpoint:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })

        # index, prediction, GT, logit
        store_model_preds = {}
        store_model_preds['indx'] = preds_and_output[:, 0]
        store_model_preds['prediction'] = preds_and_output[:, 1]
        store_model_preds['ground truth'] = preds_and_output[:, 2]
        store_model_preds['logit'] = preds_and_output[:, 3]

        return acc, store_model_preds, self.model, self.optimizer, input_images, old_outputs, prev_preds

    def training_participants(self, epoch, save_checkpoint=False, output_dir=None, round=None):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self.model.train()
        if self.args.att_guided:
            self.att_model.train()

        # initializes cool bar for visualization
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        if self.args.dataset == 'RCT':
            preds_and_output = np.array([], dtype=np.int64).reshape(0, 24)
        else:
            preds_and_output = np.array([], dtype=np.int64).reshape(0, 5)
        input_images = torch.tensor([]).cuda()
        old_outputs = torch.tensor([]).cuda()
        prev_preds = torch.tensor([]).cuda()
        # iterate over all samples in each batch i
        for i, (image, target, idxs, id) in enumerate(tbar):
            # convert target to one hot vectors
            one_hot = torch.zeros(target.shape[0], self.nclasses)
            one_hot[range(target.shape[0]), target.long()] = 1
            # assign each image and target to GPU
            if self.args.cuda:
                image, target = image.to(DEVICE), target.to(DEVICE),
                one_hot = one_hot.cuda()
                if self.args.att_guided:
                    id = id.to(DEVICE)

            # plot_batch(image, DEVICE)

            # update model
            self.optimizer.zero_grad()

            if self.args.att_guided:
                self.att_optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self.model(image)

            if self.args.att_guided:
                att_output = self.att_model(id.float())
                attpred_softmax = torch.log_softmax(att_output, dim=1)
                _, att_pred = torch.max(attpred_softmax, 1)


            logit, pred = torch.max(output.data, 1)
            total += target.size(0)

            input_images = torch.cat((input_images, image), 0)
            old_outputs = torch.cat((old_outputs, output), 0)
            prev_preds = torch.cat((prev_preds, pred), 0)

            t1 = np.expand_dims(idxs.cpu().numpy(), axis=1)
            if self.args.dataset == 'RCT':
                t2 = id.cpu().numpy()
            else:
                t2 = np.expand_dims(id.cpu().numpy(), axis=1)
            t3 = np.expand_dims(pred.cpu().numpy(), axis=1)
            t4 = np.expand_dims(target.cpu().numpy(), axis=1)
            t5 = np.expand_dims(logit.cpu().numpy(), axis=1)
            collect_preds_and_output = np.concatenate([t1, t2, t3, t4, t5], axis=1)
            preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])

            # collect forgetting events
            acc = pred.eq(target.data)
            delta = np.clip(self.prev_acc[idxs] - acc.cpu().numpy(), a_min=0, a_max=1)
            self.forgetting_events[idxs] += delta

            # mark learned samples
            self.learned[idxs] += acc.cpu().numpy()

            # update previous accuracy
            self.prev_acc[idxs] = acc.cpu().numpy()

            # Perform model update
            # calculate loss
            if self.args.train_type == 'positive_congruent' and round > 0:
                self.criterion = PositiveCongruentLoss
                OldOutputPath = os.path.join(self.args.base_dir, self.args.dataset + self.args.eval, self.args.architecture,
                                             self.args.start_strategy, self.args.out_dir,
                                             "pytorch_ckpt_" + str(round - 1) + ".tar")
                # oldModel = self.get_model(self.args.architecture)
                # oldModel = torch.nn.DataParallel(oldModel, device_ids=self.args.gpu_ids)
                # oldModel = oldModel.to(DEVICE)
                state_dict = torch.load(OldOutputPath)
                old_images = state_dict['input_images']
                old_logits = state_dict['old_output']
                old_preds = state_dict['old_preds']
                # print(old_logits.shape)
                old_logits = torch.split(old_logits, self.args.batch_size)
                old_images = torch.split(old_images, self.args.batch_size)
                old_preds = torch.split(old_preds, self.args.batch_size)
                # oldModel.load_state_dict(state_dict['model_state_dict'])
                # oldModel.eval()
                if i < len(old_images):
                    # print('hello')
                    loss = self.criterion(output, target, pred, self.model, old_logits[i], old_images[i], old_preds[i])
                else:
                    # print('byeeee')
                    CE_loss = nn.CrossEntropyLoss()
                    loss = CE_loss(output, target)
            elif self.args.att_guided:
                CE_loss = nn.CrossEntropyLoss()
                loss1 = self.criterion(output, one_hot)
                loss2 = self.criterion(att_output, one_hot)
                loss3 = self.att_criterion(output[att_pred == target], att_output[att_pred == target])
                if torch.isnan(loss3):
                    loss3 = self.att_criterion(output[pred == target], att_output[pred == target])
                loss = loss1 + loss2 + loss3
                # print('Look here -> ' + str(loss.item()))
                # print(loss.item)
            else:
                if self.args.strategy == 'badge':
                    loss = self.criterion(output, target)
                else:
                    loss = self.criterion(output, one_hot)
            # perform backpropagation
            loss.backward()

            # update params with gradients
            self.optimizer.step()
            if self.args.att_guided:
                self.att_optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if self.args.att_guided:
                correct_pred1 = pred.eq(target.data)
                correct_pred2 = att_pred.eq(target)
                correct_samples += (correct_pred1 | correct_pred2).cpu().sum()
            else:
                correct_samples += pred.eq(target.data).cpu().sum()

        # Update optimizer step
        if self.args.optimizer == 'sgd':
            self.scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        # save checkpoint
        if save_checkpoint:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })

        # if acc > self.args.min_acc:
        # index, patient ID, prediction, GT, logit
        # preds_and_output = np.asarray(preds_and_output)
        # print(round)
        # print(preds_and_output.shape)

        store_model_preds = {}
        if self.args.dataset == 'RCT':
            store_model_preds['indx'] = preds_and_output[:, 0]
            store_model_preds['id'] = preds_and_output[:, 1:21]
            store_model_preds['prediction'] = preds_and_output[:, 21]
            store_model_preds['ground truth'] = preds_and_output[:, 22]
            store_model_preds['logit'] = preds_and_output[:, 23]
            # store_model_preds['loss'] = train_loss
        else:
            store_model_preds['indx'] = preds_and_output[:, 0]
            store_model_preds['id'] = preds_and_output[:, 1]
            store_model_preds['prediction'] = preds_and_output[:, 2]
            store_model_preds['ground truth'] = preds_and_output[:, 3]
            store_model_preds['logit'] = preds_and_output[:, 4]

        return acc, store_model_preds, self.model, self.optimizer, input_images, old_outputs, prev_preds
        # else:
        #     return acc, None

    # original
    def testing_original(self, corrupted=False):
        # set model to evaluation mode
        self.model.eval()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        if not corrupted:
            tbar = tqdm(self.test_loader, desc='\r')
        else:
            if self.corr_loader is not None:
                tbar = tqdm(self.corr_loader, desc='\r')
            else:
                raise Exception("corruption loader is not defined!!!!!")

        # init statistics parameters
        test_loss = 0.0

        # overall test accuracy
        correct_samples = 0
        total_samples = 0

        # weighted test accuracy
        correct_weighted_samples = 0
        total_weighted_samples = 0
        correct_weighted_switchsamples = 0
        total_weighted_switchsamples = 0

        # difficult split accuracy
        correct_difficult = 0
        total_difficult = 0

        # easy split accuracy
        correct_easy = 0
        total_easy = 0

        difficulty_threshold = self.args.difficulty_threshold

        # iterate over all sample batches
        for i, (image, target, idxs) in enumerate(tbar):
            # convert target to one hot vectors

            one_hot = torch.zeros(target.shape[0], self.nclasses)
            one_hot[range(target.shape[0]), target.long()] = 1
            # set cuda
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                one_hot = one_hot.cuda()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():
                # get output
                output = self.model(image)
                # calculate loss between output and target
                if self.args.strategy == 'badge':
                    loss = self.criterion(output, target.long())
                else:
                    loss = self.criterion(output, one_hot)
                # append loss to total loss
                test_loss += loss.item()

                _, pred = torch.max(output.data, 1)

                if not corrupted:
                    # collect forgetting events
                    acc = pred.eq(target.data)
                    delta = np.clip(self.test_prev_acc[idxs] - acc.cpu().numpy(), a_min=0, a_max=1)
                    self.test_forgetting_events[idxs] += delta

                    # mark learned samples
                    self.test_learned[idxs] += acc.cpu().numpy()

                    # update previous accuracy
                    self.test_prev_acc[idxs] = acc.cpu().numpy()

                if self.weight_eval and not corrupted:
                    # get weight according to forgetfulness -> shift by one so unforgettable samples count as well
                    fevents_weight = self.test_weights[idxs] + 1

                    # weighted acc
                    total_weighted_samples += target.size(0) * fevents_weight
                    correct_weighted_samples += (pred.eq(target.data).cpu().sum()) * fevents_weight
                    total_weighted_switchsamples += target.size(0) * self.test_switch_weights[idxs]
                    correct_weighted_switchsamples += (pred.eq(target.data).cpu().sum()) * self.test_switch_weights[idxs]

                    # difficult acc
                    if self.test_weights[idxs] < difficulty_threshold:
                        total_easy += target.size(0)
                        correct_easy += pred.eq(target.data).cpu().sum()
                    else:
                        total_difficult += target.size(0)
                        correct_difficult += pred.eq(target.data).cpu().sum()

                # overall acc
                total_samples += target.size(0)
                correct_samples += pred.eq(target.data).cpu().sum()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total_samples

        # Print statistics
        print('Testing:')
        print('Loss: %.3f' % test_loss)
        print('Test Accuracy: %.3f' % acc)
        if self.weight_eval and not corrupted:
            weighted_acc = 100.0 * correct_weighted_samples.item() / total_weighted_samples
            weighted_switch_acc = 100.0 * correct_weighted_switchsamples.item() / total_weighted_switchsamples
            if total_difficult != 0:
                difficult_acc = 100.0 * correct_difficult.item() / total_difficult
            else:
                difficult_acc = -1.0

            if total_easy != 0:
                easy_acc = 100.0 * correct_easy.item() / total_easy
            else:
                easy_acc = -1.0
            print('EF Test Accuracy: %.3f' % weighted_acc)
            print('Switch Test Accuracy: %.3f' % weighted_switch_acc)
            print('Easy Test Accuracy: %.3f' % easy_acc)
            print('Difficult Test Accuracy: %.3f' % difficult_acc)
        else:
            weighted_acc = -1
            easy_acc = -1
            difficult_acc = -1
            weighted_switch_acc = -1
        output_structure = {}
        output_structure['test_acc'] = acc
        output_structure['ef_acc'] = weighted_acc
        output_structure['switch_acc'] = weighted_switch_acc
        output_structure['easy_acc'] = easy_acc
        output_structure['difficult_acc'] = difficult_acc
        return output_structure

    def testing(self, corrupted=False, round=None):
        # set model to evaluation mode
        self.model.eval()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        if not corrupted:
            tbar = tqdm(self.test_loader, desc='\r')
        else:
            if self.corr_loader is not None:
                tbar = tqdm(self.corr_loader, desc='\r')
            else:
                raise Exception("corruption loader is not defined!!!!!")

        # init statistics parameters
        test_loss = 0.0
        gt_patient_count = {}  # the total number of times a patient occurs with its gt label
        pred_patient_label_count = {}  # the total number of times that patient is predicted correctly
        gt_labels = {}

        # overall test accuracy
        correct_samples = 0
        total_samples = 0

        # weighted test accuracy
        correct_weighted_samples = 0
        total_weighted_samples = 0
        correct_weighted_switchsamples = 0
        total_weighted_switchsamples = 0

        # difficult split accuracy
        correct_difficult = 0
        total_difficult = 0

        # easy split accuracy
        correct_easy = 0
        total_easy = 0

        difficulty_threshold = self.args.difficulty_threshold

        preds_and_output = np.array([], dtype=np.int64).reshape(0, 4)
        # Iterate over all sample batches
        for i, (image, target, idxs) in enumerate(tbar):
            # patient ID
            # if id.item() not in gt_patient_count.keys():
            #     gt_patient_count[id.item()] = 1
            #     gt_labels[id.item()] = target.item()
            # else:
            #     gt_patient_count[id.item()] += 1

            # convert target to one hot vectors
            # one_hot = torch.zeros(target.shape[0], self.nclasses)
            # one_hot[range(target.shape[0]), target.long()] = 1
            # set cuda
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                # one_hot = one_hot.cuda()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():
                # get output
                output = self.model(image)
                logit, pred = torch.max(output.data, 1)

                # calculate loss between output and target
                # if self.args.train_type == 'positive_congruent' and round > 0:
                #     self.criterion = PositiveCongruentLoss
                #     OldOutputPath = os.path.join(self.args.base_dir, self.args.dataset + self.args.eval, self.args.architecture,
                #                                  self.args.start_strategy, self.args.out_dir,
                #                                  "positive_cong_test_info" + str(round - 1) + ".xlsx")
                #     old_df = pd.read_excel(OldOutputPath, index_col=0, engine='openpyxl')
                #     old_logits = old_df.loc[:, 'logit'].to_numpy()
                #     old_preds = old_df.loc[:, 'prediction'].to_numpy()
                #     loss = self.criterion(output, target, pred, self.model, old_logits[i], image, old_preds[i], mode='test')
                # else:
                #     if self.args.strategy == 'badge':
                #         loss = self.criterion(output, target.long())
                #     else:
                #         temp = nn.CrossEntropyLoss()
                #         loss = temp(output, target)
                        # loss = self.criterion(output, one_hot)  # original
                # append loss to total loss
                # test_loss += loss.item()

                # if pred == target:
                #     if id.item() not in pred_patient_label_count.keys():
                #         pred_patient_label_count[id.item()] = 1
                #     else:
                #         pred_patient_label_count[id.item()] += 1
                # else:
                #     if id.item() not in pred_patient_label_count.keys():
                #         pred_patient_label_count[id.item()] = 0

                # index, prediction, GT, logit
                # preds_and_output.append([idxs.item(), pred.item(), target.item(), output.data[0, pred.item()].item()])
                t1 = np.expand_dims(idxs.cpu().numpy(), axis=1)
                t3 = np.expand_dims(pred.cpu().numpy(), axis=1)
                t4 = np.expand_dims(target.cpu().numpy(), axis=1)
                t5 = np.expand_dims(logit.cpu().numpy(), axis=1)
                # t5 = output.data.gather(1, pred.view(-1, 1)).cpu().numpy()
                collect_preds_and_output = np.concatenate([t1, t3, t4, t5], axis=1)
                preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])

                if not corrupted:
                    # collect forgetting events
                    acc = pred.eq(target.data)
                    delta = np.clip(self.test_prev_acc[idxs] - acc.cpu().numpy(), a_min=0, a_max=1)
                    self.test_forgetting_events[idxs] += delta

                    # mark learned samples
                    self.test_learned[idxs] += acc.cpu().numpy()

                    # update previous accuracy
                    self.test_prev_acc[idxs] = acc.cpu().numpy()

                if self.weight_eval and not corrupted:
                    # get weight according to forgetfulness -> shift by one so unforgettable samples count as well
                    fevents_weight = self.test_weights[idxs] + 1

                    # weighted acc
                    total_weighted_samples += target.size(0) * fevents_weight
                    correct_weighted_samples += (pred.eq(target.data).cpu().sum()) * fevents_weight
                    total_weighted_switchsamples += target.size(0) * self.test_switch_weights[idxs]
                    correct_weighted_switchsamples += (pred.eq(target.data).cpu().sum()) * self.test_switch_weights[idxs]

                    # difficult acc
                    if self.test_weights[idxs] < difficulty_threshold:
                        total_easy += target.size(0)
                        correct_easy += pred.eq(target.data).cpu().sum()
                    else:
                        total_difficult += target.size(0)
                        correct_difficult += pred.eq(target.data).cpu().sum()

                # overall acc
                total_samples += target.size(0)
                correct_samples += pred.eq(target.data).cpu().sum()

            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total_samples

        # Print statistics
        print('Testing:')
        # print('Loss: %.3f' % test_loss)
        print('Test Accuracy: %.3f' % acc)
        if self.weight_eval and not corrupted:
            weighted_acc = 100.0 * correct_weighted_samples.item() / total_weighted_samples
            weighted_switch_acc = 100.0 * correct_weighted_switchsamples.item() / total_weighted_switchsamples
            if total_difficult != 0:
                difficult_acc = 100.0 * correct_difficult.item() / total_difficult
            else:
                difficult_acc = -1.0

            if total_easy != 0:
                easy_acc = 100.0 * correct_easy.item() / total_easy
            else:
                easy_acc = -1.0
            print('EF Test Accuracy: %.3f' % weighted_acc)
            print('Switch Test Accuracy: %.3f' % weighted_switch_acc)
            print('Easy Test Accuracy: %.3f' % easy_acc)
            print('Difficult Test Accuracy: %.3f' % difficult_acc)
        else:
            weighted_acc = -1
            easy_acc = -1
            difficult_acc = -1
            weighted_switch_acc = -1
        output_structure = {}
        output_structure['test_acc'] = acc
        output_structure['ef_acc'] = weighted_acc
        output_structure['switch_acc'] = weighted_switch_acc
        output_structure['easy_acc'] = easy_acc
        output_structure['difficult_acc'] = difficult_acc
        output_structure['pred_patient_label_count'] = pred_patient_label_count
        output_structure['gt_patient_count'] = gt_patient_count
        output_structure['gt_labels'] = gt_labels


        # index, prediction, GT, logit
        # preds_and_output = np.asarray(preds_and_output)
        store_model_preds = {}
        store_model_preds['indx'] = preds_and_output[:, 0]
        store_model_preds['prediction'] = preds_and_output[:, 1]
        store_model_preds['ground truth'] = preds_and_output[:, 2]
        store_model_preds['logit'] = preds_and_output[:, 3]

        return output_structure, store_model_preds

    def testing_participants(self, corrupted=False, round=None):
        # set model to evaluation mode
        self.model.eval()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        if not corrupted:
            tbar = tqdm(self.test_loader, desc='\r')
        else:
            if self.corr_loader is not None:
                tbar = tqdm(self.corr_loader, desc='\r')
            else:
                raise Exception("corruption loader is not defined!!!!!")

        # init statistics parameters
        test_loss = 0.0
        gt_patient_count = {}  # the total number of times a patient occurs with its gt label
        pred_patient_label_count = {}  # the total number of times that patient is predicted correctly
        gt_labels = {}

        # overall test accuracy
        correct_samples = 0
        total_samples = 0

        pred_all = np.array([], dtype=np.int)
        gt_all = np.array([], dtype=np.int)


        # weighted test accuracy
        correct_weighted_samples = 0
        total_weighted_samples = 0
        correct_weighted_switchsamples = 0
        total_weighted_switchsamples = 0

        # difficult split accuracy
        correct_difficult = 0
        total_difficult = 0

        # easy split accuracy
        correct_easy = 0
        total_easy = 0

        difficulty_threshold = self.args.difficulty_threshold

        preds_and_output = []

        # Iterate over all sample batches
        for i, (image, target, idxs, id) in enumerate(tbar):
            # patient ID
            if id.item() not in gt_patient_count.keys():
                gt_patient_count[id.item()] = 1
                gt_labels[id.item()] = target.item()
            else:
                gt_patient_count[id.item()] += 1

            # convert target to one hot vectors
            one_hot = torch.zeros(target.shape[0], self.nclasses)
            one_hot[range(target.shape[0]), target.long()] = 1
            # set cuda
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                one_hot = one_hot.cuda()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():
                # get output
                output = self.model(image)
                _, pred = torch.max(output.data, 1)

                pred_all = np.concatenate((pred_all, pred.cpu().numpy()), axis=0)
                gt_all = np.concatenate((gt_all, target.cpu().numpy()), axis=0)

                # calculate loss between output and target
                if self.args.train_type == 'positive_congruent' and round > 0:
                    self.criterion = PositiveCongruentLoss
                    OldOutputPath = os.path.join(self.args.base_dir, self.args.dataset + self.args.eval, self.args.architecture,
                                                 self.args.start_strategy, self.args.out_dir,
                                                 "positive_cong_test_info" + str(round - 1) + ".xlsx")
                    old_df = pd.read_excel(OldOutputPath, index_col=0, engine='openpyxl')
                    old_logits = old_df.loc[:, 'logit'].to_numpy()
                    old_preds = old_df.loc[:, 'prediction'].to_numpy()
                    # oldModel = self.get_model(self.args.architecture)
                    # oldModel = torch.nn.DataParallel(oldModel, device_ids=self.args.gpu_ids)
                    # oldModel = oldModel.to(DEVICE)
                    # state_dict = torch.load(OldOutputPath)
                    # oldModel.load_state_dict(state_dict['model_state_dict'])
                    # oldModel.eval()
                    # old_logits = oldModel(image)
                    loss = self.criterion(output, target, pred, self.model, old_logits[i], image, old_preds[i], mode='test')
                else:
                    if self.args.strategy == 'badge':
                        loss = self.criterion(output, target.long())
                    else:
                        loss = self.criterion(output, one_hot)
                # append loss to total loss
                test_loss += loss.item()

                if pred == target:
                    if id.item() not in pred_patient_label_count.keys():
                        pred_patient_label_count[id.item()] = 1
                    else:
                        pred_patient_label_count[id.item()] += 1
                else:
                    if id.item() not in pred_patient_label_count.keys():
                        pred_patient_label_count[id.item()] = 0

                # index, patient ID, prediction, GT, logit
                preds_and_output.append([idxs.item(), id.item(), pred.item(), target.item(), output.data[0, pred.item()].item()])

                if not corrupted:
                    # collect forgetting events
                    acc = pred.eq(target.data)
                    delta = np.clip(self.test_prev_acc[idxs] - acc.cpu().numpy(), a_min=0, a_max=1)
                    self.test_forgetting_events[idxs] += delta

                    # mark learned samples
                    self.test_learned[idxs] += acc.cpu().numpy()

                    # update previous accuracy
                    self.test_prev_acc[idxs] = acc.cpu().numpy()

                if self.weight_eval and not corrupted:
                    # get weight according to forgetfulness -> shift by one so unforgettable samples count as well
                    fevents_weight = self.test_weights[idxs] + 1

                    # weighted acc
                    total_weighted_samples += target.size(0) * fevents_weight
                    correct_weighted_samples += (pred.eq(target.data).cpu().sum()) * fevents_weight
                    total_weighted_switchsamples += target.size(0) * self.test_switch_weights[idxs]
                    correct_weighted_switchsamples += (pred.eq(target.data).cpu().sum()) * self.test_switch_weights[idxs]

                    # difficult acc
                    if self.test_weights[idxs] < difficulty_threshold:
                        total_easy += target.size(0)
                        correct_easy += pred.eq(target.data).cpu().sum()
                    else:
                        total_difficult += target.size(0)
                        correct_difficult += pred.eq(target.data).cpu().sum()

                # overall acc
                total_samples += target.size(0)
                correct_samples += pred.eq(target.data).cpu().sum()
                # precision = precision_score(target.data.cpu().numpy(), pred.cpu().numpy(), average=None)
                # recall = recall_score(target.data.cpu().numpy(), pred.cpu().numpy(), average=None)
                # precion_all = np.concatenate((precion_all, precision), axis=0)
                # recall_all = np.concatenate((recall_all, recall), axis=0)

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total_samples
        precision = precision_score(gt_all, pred_all, average=None)
        recall = recall_score(gt_all, pred_all, average=None)


        # Print statistics
        print('Testing:')
        print('Loss: %.3f' % test_loss)
        print('Test Accuracy: %.3f' % acc)
        if self.weight_eval and not corrupted:
            weighted_acc = 100.0 * correct_weighted_samples.item() / total_weighted_samples
            weighted_switch_acc = 100.0 * correct_weighted_switchsamples.item() / total_weighted_switchsamples
            if total_difficult != 0:
                difficult_acc = 100.0 * correct_difficult.item() / total_difficult
            else:
                difficult_acc = -1.0

            if total_easy != 0:
                easy_acc = 100.0 * correct_easy.item() / total_easy
            else:
                easy_acc = -1.0
            print('EF Test Accuracy: %.3f' % weighted_acc)
            print('Switch Test Accuracy: %.3f' % weighted_switch_acc)
            print('Easy Test Accuracy: %.3f' % easy_acc)
            print('Difficult Test Accuracy: %.3f' % difficult_acc)
        else:
            weighted_acc = -1
            easy_acc = -1
            difficult_acc = -1
            weighted_switch_acc = -1
        output_structure = {}
        output_structure['test_acc'] = acc
        output_structure['test_precision'] = precision
        output_structure['test_recall'] = recall
        output_structure['ef_acc'] = weighted_acc
        output_structure['switch_acc'] = weighted_switch_acc
        output_structure['easy_acc'] = easy_acc
        output_structure['difficult_acc'] = difficult_acc
        output_structure['pred_patient_label_count'] = pred_patient_label_count
        output_structure['gt_patient_count'] = gt_patient_count
        output_structure['gt_labels'] = gt_labels


        # index, patient ID, prediction, GT, logit
        preds_and_output = np.asarray(preds_and_output)
        store_model_preds = {}
        store_model_preds['indx'] = preds_and_output[:, 0]
        store_model_preds['id'] = preds_and_output[:, 1]
        store_model_preds['prediction'] = preds_and_output[:, 2]
        store_model_preds['ground truth'] = preds_and_output[:, 3]
        store_model_preds['logit'] = preds_and_output[:, 4]

        return output_structure, store_model_preds

    def get_probs(self):
        '''Calculates the gradients for all elements within the test pool and ranks the highest idxs'''
        # update gradnorm update number
        # self.num_gradnorm_updates += 1
        # set model to evaluation mode
        self.model.eval()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        tbar = tqdm(self.grad_loader, desc='\r')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init probability array to zero maximum value of sigmoid is 1.0 therefore ignore all values larger than that
        probs = torch.full((self.train_pool, self.nclasses), 2.5, dtype=torch.float)
        indices = torch.zeros(self.train_pool)
        if self.args.dataset == 'RCT':
            possible_bioindicators = 20
            ids = torch.zeros((self.train_pool, possible_bioindicators), dtype=torch.long)
        else:
            ids = torch.zeros(self.train_pool, dtype=torch.long)

        if self.args.cuda:
            probs, indices = probs.to(DEVICE), indices.to(DEVICE)

        with torch.no_grad():
            # iterate over all sample batches
            for i, (image, target, idxs, id) in enumerate(tbar):
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(DEVICE), target.to(DEVICE)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                # get sigmoid probs
                probs_output = softmax(output)

                # insert to probs array
                probs[idxs.long()] = probs_output
                indices[idxs.long()] = 1
                ids[idxs.long()] = id

        # sort idxs
        output_structure = {}
        output_structure['probs'] = probs[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        ID_list = ids[indices == 1].cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['IDs'] = ID_list

        return output_structure

    def get_badge_embeddings(self):
        '''Calculates the gradients for all elements within the test pool and ranks the highest idxs'''
        # update gradnorm update number
        # self.num_gradnorm_updates += 1
        # set model to evaluation mode
        self.model.eval()

        # get embed dim

        if self.args.cuda:
            embedDim = self.model.module.get_penultimate_dim()
        else:
            embedDim = self.model.get_penultimate_dim()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        tbar = tqdm(self.grad_loader, desc='\r')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.train_pool, embedDim * self.nclasses), dtype=torch.float)
        indices = torch.zeros(self.train_pool)

        if self.args.dataset == 'RCT':
            possible_bioindicators = 20
            ids = torch.zeros((self.train_pool, possible_bioindicators), dtype=torch.long)
        else:
            ids = torch.zeros(self.train_pool, dtype=torch.long)

        if self.args.cuda:
            embeddings, indices = embeddings.to(DEVICE), indices.to(DEVICE)

        with torch.no_grad():
            # iterate over all sample batches
            for i, (image, target, idxs, id) in enumerate(tbar):
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(DEVICE), target.to(DEVICE)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                # get penultimate embedding
                if self.args.cuda:
                    penultimate = self.model.module.penultimate_layer
                else:
                    penultimate = self.model.penultimate_layer

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embediing array
                for j in range(target.shape[0]):
                    for c in range(self.nclasses):
                        if c == pred[j].item():
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (1 - probs_output[j, c].item())
                        else:
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (-1 * probs_output[j, c].item())
                indices[idxs.long()] = 1
                ids[idxs.long()] = id

        # sort idxs
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        ID_list = ids[indices == 1].cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['IDs'] = ID_list

        return output_structure

    def track_unlabeled_forgetting_statistics(self, epoch):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self.model.eval()
        # initializes cool bar for visualization
        tbar = tqdm(self.unlabeled_loader)
        num_img_tr = len(self.unlabeled_loader)

        # init statistics parameters
        correct_samples = 0
        total = 0
        switches = 0

        with torch.no_grad():
            # iterate over all samples in each batch i
            for i, (image, target, idxs) in enumerate(tbar):
                # convert target to one hot vectors
                one_hot = torch.zeros(target.shape[0], self.nclasses)
                one_hot[range(target.shape[0]), target.long()] = 1
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(DEVICE), target.to(DEVICE)
                    one_hot = one_hot.cuda()

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                _, pred = torch.max(output.data, 1)
                total += target.size(0)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                acc = self.unlabeled_prev_pred[idxs] != predicted

                # only track if epoch is larger thatn zero otherwirse first epoch and previous accuracy does not exist
                if epoch > 0:
                    self.unlabeled_variety[idxs] += acc

                # mark learned samples
                self.unlabeled_seen[idxs] = 1

                # update previous accuracy
                self.unlabeled_prev_pred[idxs] = predicted

                # Do fun bar stuff
                switches = switches + np.count_nonzero(acc)
                tbar.set_description('Number of switches: %d' % switches)

                correct_samples += pred.eq(target.data).cpu().sum()

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        print('Unlabeled Accuracy: %.3f' % acc)
        return switches

    def save_statistics(self, save_path, round=0):
        # init samples that were never learned as max forgetting events + 1
        largest = max(self.forgetting_events)
        self.forgetting_events[self.learned == 0] = largest + 1
        test_largest = max(self.test_forgetting_events)
        self.test_forgetting_events[self.test_learned == 0] = test_largest + 1
        # get names and save
        learned_name = os.path.join(save_path, 'learned_' + str(round) + '.npy')
        forgetting_name = os.path.join(save_path, 'forgetting_events' + str(round) + '.npy')
        test_learned_name = os.path.join(save_path, 'test_learned_' + str(round) + '.npy')
        test_forgetting_name = os.path.join(save_path, 'test_forgetting_events' + str(round) + '.npy')
        gradnorm_name = os.path.join(save_path, 'sorted_mcuncertainties' + str(round) + '.npy')
        gradindex_name = os.path.join(save_path, 'sorted_mc_index' + str(round) + '.npy')
        np.save(forgetting_name, self.forgetting_events)
        np.save(learned_name, self.learned)
        np.save(test_forgetting_name, self.test_forgetting_events)
        np.save(test_learned_name, self.test_learned)

        # gradnorms and indexes separately
        if self.args.strategy == 'mc_update':
            gr_norms = np.array(sorted(self.mc_uncertainties.values()))
            np.save(gradnorm_name, gr_norms)
            gr_indexes = np.array(self.sorted_mclist)
            np.save(gradindex_name, gr_indexes)

        # save unlabeled indices only if calculated
        unlabeled = self.unlabeled_variety[self.unlabeled_seen == 1]
        unlabeled_inds = np.where(self.unlabeled_seen == 1)[0]
        unlabeled_name = os.path.join(save_path, 'unlabeled_switching_frequency' + str(round) + '.npy')
        inds_name = os.path.join(save_path, 'unlabeled_switching_indices' + str(round) + '.npy')
        np.save(unlabeled_name, unlabeled)
        np.save(inds_name, unlabeled_inds)

    def get_embeddings(self, loader_type: str = 'unlabeled'):
        # set model to evaluation mode
        self._model.eval()

        # get embed dim
        if self.args.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        # give me more baaaaaaaaaaaaaaaaaaaaaaaaaar!!!
        if loader_type == 'labeled':
            tbar = tqdm.tqdm(self.train_loader, desc='\r')
        elif loader_type == 'unlabeled':
            tbar = tqdm.tqdm(self.unlabeled_loader, desc='\r')
        else:
            raise Exception('You can only load labeled and unlabeled pools!')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.train_pool, embedDim * self.nclasses),
                                 dtype=torch.float)
        nongrad_embeddings = torch.zeros((self.train_pool, embedDim),
                                         dtype=torch.float)
        indices = torch.zeros(self.train_pool)

        if self.args.cuda:
            embeddings, indices = embeddings.to(DEVICE), indices.to(DEVICE)
            nongrad_embeddings = nongrad_embeddings.to(DEVICE)

        with torch.no_grad():
            # iterate over all sample batches
            for i, sample in enumerate(tbar):
                image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(DEVICE), target.to(DEVICE)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self._model(image)

                # get penultimate embedding
                if self.args.cuda:
                    penultimate = self._model.module.penultimate_layer
                else:
                    penultimate = self._model.penultimate_layer

                nongrad_embeddings[idxs] = penultimate

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embediing array
                for j in range(target.shape[0]):
                    for c in range(self.nclasses):
                        if c == pred[j].item():
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (1 - probs_output[j, c].item())
                        else:
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(penultimate[j]) * \
                                                                                    (-1 * probs_output[j, c].item())
                indices[idxs.long()] = 1

        # sort idxs
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['nongrad_embeddings'] = nongrad_embeddings[indices == 1].cpu().numpy()

        return output_structure