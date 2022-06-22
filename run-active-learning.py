import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import shutil
import torch
import pandas as pd
from mypath import Path, machine
import numpy as np
from collections import Counter
from myUtils.trainer import Trainer
from query_strategies import RandomSampling, IdealEFSampling, LeastConfidenceSampler, EntropySampler, \
    MarginSampler, BadgeSampler, SwitchSampling, PatientDiverseSampler, \
    PatientDiverseEntropySampler, PatientDiverseEntropyMacroSampler, PatientDiverseMarginSampler, \
    PatientDiverseLeastConfidenceSampler, PatientDiverseBadgeSampler, ClinicallyDiverseSampler, \
    ClinicallyDiverseEntropySampler, ClinicallyDiverseBadgeSampler


def parse_everything():
    parser = argparse.ArgumentParser(description="PyTorch Forgetting events classification Training")
    parser.add_argument('--architecture', type=str, default='resnet_18',
                        choices=['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152',
                                 'densenet_121', 'densenet_161', 'densenet_169', 'densenet_201',
                                 'vgg_11', 'vgg_13', 'vgg_16', 'vgg_19', 'mlp'],
                        help='architecture name (default: resnet)')
    parser.add_argument('--exp_type', type=str, default=None, choices=['initialization'], help='runs only round 0')
    parser.add_argument('--dataset', type=str, default='RCT',  ############## TODO!!!!!!!!
                        choices=['CIFAR10', 'STL10', 'MNIST', 'SVHN', 'Kermany', 'KermanyXray', 'RCT'],
                        help='dataset name (default: Kermany)')
    parser.add_argument('--download', type=bool, default=False,  ##########
                        help='specifies whether to download the dataset or not (default: False)')
    parser.add_argument('--data_path', type=str,
                        default='./Data',
                        help='dataset path')
    parser.add_argument('--train_type', type=str, default='traditional',  # TODO
                        choices=['traditional', 'positive_congruent'])
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--run_status', type=str, default='train',
                        choices=['train', 'test'], help='seimsic test type on data type')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=128,  ######## 2
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,  #TODO!!!!!!!
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--unlabeled_batch_size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--optimizer', default="adam",
                        help='optimizer to use, default is sgd. Can also use adam')  #### sgd
    parser.add_argument('--lr', type=float, default=0.00015,  ####### TODO!!!!! 0.001 CIFAR-10, 0.00015 kermany
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')

    # output directory for learning maps
    parser.add_argument('--out_dir', type=str, default='output_default/',
                        help='path to output directory for output files (excel, learning maps)')
    parser.add_argument('--recording_epoch', type=int, default=40,
                        help='evaluation interval (default: 1)')

    # active learning parameters
    parser.add_argument('--strategy', type=str, default='clinically_diverse_entropy',  ######### rand
                        choices=['rand', 'idealEF', 'reversed_idealEF', 'mc_update', 'least_conf', 'entropy', 'margin',
                                 'badge', 'gradcon', 'reversed_EF_switchsampling', 'EF_switchsampling', 'idealBadge',
                                 'patient_diverse', 'patient_diverse_entropy', 'patient_diverse_entropy_macro',
                                 'patient_diverse_margin', 'patient_diverse_least_conf', 'patient_diverse_badge',
                                 'clinically_diverse', 'clinically_diverse_entropy', 'clinically_diverse_badge'],
                        help='strategy used for sample query in active earning experiment')
    parser.add_argument('--start_strategy', type=str, default='diverse_init',  # TODO!!!!!
                        choices=['rand_init', 'diverse_init'])  ###########
    parser.add_argument('--nstart', type=int, default=128,  ######### 20 # TODO
                        help='number of samples in the initial data pool')
    parser.add_argument('--nend', type=int, default=3000,  ######## 50000
                        help='maximum amount of points to be queried')
    parser.add_argument('--nquery', type=int, default=128,  ######## 1024 cifar-10 TODO
                        help='number of samples to be queried in each round')
    parser.add_argument('--min_acc', type=float, default=98.0,  # TODO
                        help='number of samples to be queried in each round')

    # example forgetting parameters
    parser.add_argument('--ef_index_path', type=str,
                        default='example_forgetting_statistics-resnet18-80epochs-CIFAR10/sorted_indexes.npy',
                        help='path for fevents strategy')
    parser.add_argument('--ef_eval_path', type=str,
                        default=None,
                        help='path for fevents evaluation in the test set')
    parser.add_argument('--track_test_fevents', action='store_true', default=False,
                        help='flag for tracking gradients')
    parser.add_argument('--eval_test_epoch', type=int, default=10,
                        help='epoch tracking rate for forgetting test events. Smaller more exact but harder compute.')
    parser.add_argument('--difficulty_threshold', type=int, default=1,
                        help='difficulty threshold for difficult or easy classification.')

    # switch sampling parameters
    parser.add_argument('--switch_eval_path', type=str,
                        default=None,
                        help='path for fevents evaluation in the test set')

    # switch sampling parameters
    parser.add_argument('--corruption', type=str,
                        default=None,
                        help='corruption type for testing')
    parser.add_argument('--track-switch-events', type=bool, default=False,
                        help='specifies whether to track prediction switches on unlabeled data (default: False)')
    parser.add_argument('--eval', type=str, default='Patient',  # TODO
                        choices=['Patient', 'Traditional'], help='Evaluate test set at patient level')  ########
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='specify if a pre-trained model is used')
    parser.add_argument('--base_dir', type=str, default='output_default')
    parser.add_argument('--att_guided', type=bool, default=False)
    parser.add_argument('--use_pseudo', type=bool, default=False)
    args = parser.parse_args()

    return args


def main():
    # parse all arguments
    args = parse_everything()
    base_dir = args.base_dir
    # make sure gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # make gpu ids inputable
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'cifar10': 75
        }
        args.epochs = epoches[args.dataset.lower()]

    # default batch size
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    # default test batch size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # default learning rate
    if args.lr is None:
        lrs = {
            'cifar10': 0.1
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    # default checkpoint name
    if args.checkname is None:
        args.checkname = args.architecture
    print(args)

    # init seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # make directory to save forgetting events data
    output_directory = os.path.join(base_dir, args.dataset + args.eval, args.architecture, args.start_strategy,
                                    args.out_dir)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # init trainer class with args
    trainer = Trainer(args)

    # init sampler
    train_pool = trainer.train_pool
    init_samples = args.nstart
    total = np.arange(train_pool)

    print('Total amount of samples: %d' % len(total))

    # patient diverse initialization
    if args.start_strategy == 'diverse_init':
        import random
        # indices for patients with same ID
        id_idx = dict()
        for idx, id in enumerate(trainer.unlabeled_loader.dataset.ID):
            if id not in id_idx:
                id_idx[id] = [idx]
            else:
                id_idx[id].append(idx)

        # Random list of unique IDs
        unique_IDs = random.choices(list(id_idx.keys()), k=init_samples)

        # select a random idx for each ID
        start_idxs = []
        for id in unique_IDs:
            rand_idx = random.choice(id_idx[id])
            start_idxs.append(rand_idx)
    else:
        # start indexes are chosen randomly
        start_idxs = total[np.random.permutation(len(total))][:init_samples]

    if args.strategy == 'rand':
        print('Using a random sampler')
        sampler = RandomSampling(train_pool, start_idxs)
    elif args.strategy == 'idealEF' or args.strategy == 'reversed_idealEF':
        print('Sampling from ideal forgetting events')
        if args.ef_index_path is not None:
            sorted_fevents_idx = np.load(args.ef_index_path)

            # remove start inds from the sorted fevents samples
            for i in range(init_samples):
                sorted_fevents_idx = sorted_fevents_idx[sorted_fevents_idx != start_idxs[i]]
        else:
            raise Exception('Index path must be specified when using example forgetting!!!')

        # init ef sampler
        if args.strategy == 'idealEF':
            sampler = IdealEFSampling(train_pool, start_idxs, sorted_fevents_idx, query_type='nSV')
        elif args.strategy == 'reversed_idealEF':
            sampler = IdealEFSampling(train_pool, start_idxs, sorted_fevents_idx, query_type='SV')
        else:
            raise Exception('Something went terribly wrong here')
    elif args.strategy == 'least_conf':
        print('Using least confidence sampler')
        sampler = LeastConfidenceSampler(train_pool, start_idxs)
    elif args.strategy == 'entropy':
        print('Using least entropy sampler')
        sampler = EntropySampler(train_pool, start_idxs)
    elif args.strategy == 'margin':
        print('Using least margin sampler')
        sampler = MarginSampler(train_pool, start_idxs)
    elif args.strategy == 'badge':
        print('Using badge sampler')
        sampler = BadgeSampler(train_pool, start_idxs)
    elif args.strategy == 'idealBadge':
        if args.ef_index_path is not None:
            print('Using ideal badge sampler')
        else:
            raise Exception('Badge path must be inserted!!!')
        sampler = BadgeSampler(train_pool, start_idxs, ideal_path=args.ef_index_path)
    elif args.strategy == 'reversed_EF_switchsampling' or args.strategy == 'EF_switchsampling':
        print('Using least switch sampler')
        if args.strategy == 'reversed_EF_switchsampling':
            sampler = SwitchSampling(train_pool, start_idxs, query_type='SV')
        else:
            sampler = SwitchSampling(train_pool, start_idxs, query_type='nSV')
    elif args.strategy == 'patient_diverse':
        print('Using patient diverse random sampler')
        sampler = PatientDiverseSampler(train_pool, start_idxs)
    elif args.strategy == 'patient_diverse_entropy':
        print('Using patient diverse entropy micro sampler')
        sampler = PatientDiverseEntropySampler(train_pool, start_idxs)
    elif args.strategy == 'patient_diverse_entropy_macro':
        print('Using patient diverse entropy macro sampler')
        sampler = PatientDiverseEntropyMacroSampler(train_pool, start_idxs)
    elif args.strategy == 'patient_diverse_margin':
        print('Using patient diverse least margin sampler')
        sampler = PatientDiverseMarginSampler(train_pool, start_idxs)
    elif args.strategy == 'patient_diverse_least_conf':
        print('Using patient diverse least confidence sampler')
        sampler = PatientDiverseLeastConfidenceSampler(train_pool, start_idxs)
    elif args.strategy == 'patient_diverse_badge':
        print('Using patient diverse BADGE sampler')
        sampler = PatientDiverseBadgeSampler(train_pool, start_idxs)
    elif args.strategy == 'clinically_diverse':
        print('Using clinically diverse random sampler')
        sampler = ClinicallyDiverseSampler(train_pool, start_idxs)
    elif args.strategy == 'clinically_diverse_entropy':
        print('Using clinically diverse entropy sampler')
        sampler = ClinicallyDiverseEntropySampler(train_pool, start_idxs)
    elif args.strategy == 'clinically_diverse_badge':
        print('Using clinically diverse BADGE sampler')
        sampler = ClinicallyDiverseBadgeSampler(train_pool, start_idxs)
    else:
        print('Sampler not implemented!!!!')
        raise NotImplementedError

    # get active learning parameters
    # parameters
    NUM_QUERY = args.nquery
    if args.nend < len(total):
        NUM_ROUND = int((args.nend - init_samples) / args.nquery)
        print('Rounds: %d' % NUM_ROUND)
    else:
        NUM_ROUND = int((len(total) - init_samples) / args.nquery) + 1
        print('Number of end samples too large! Using total number of samples instead. Rounds: %d Total Samples: %d' %
              (NUM_ROUND, len(total)))

    # init dataframe
    if args.exp_type == 'initialization':
        df_path = os.path.join('excel', args.dataset, args.architecture, args.start_strategy, args.strategy + '_' + str(args.nquery))
    else:
        temp = args.out_dir.split('/')[0]
        df_path = os.path.join('excel', args.dataset, args.architecture, args.start_strategy, args.strategy + '_' + str(args.nquery))
    df_name = os.path.join(df_path, 'test_accuracy' + str(args.seed) + '.xlsx')
    if not os.path.exists(df_path):
        os.makedirs(df_path)
    df = pd.DataFrame(columns=['Samples', 'Test Acc', 'Test Precision', 'Test Recall', 'EF Test Acc', 'Easy Test Acc',
                               'Difficult Test Acc', 'Switch Test Acc', 'Corrupted Acc', 'Prediction Switches'])

    # differentaite with train and test version
    if args.run_status == 'train':
        # param for saving
        track_events = False
        # train over number of epochs
        for round in range(NUM_ROUND):
            print('Round: %d' % round)
            if args.exp_type == 'initialization' and round > 0:
                exit()

            if args.eval == 'Patient':
                dff_name = os.path.join(output_directory, 'patient_test_accuracy' + str(round) + '.xlsx')
                dff = pd.DataFrame(columns=['ID', 'Predicted Correct', 'Total Count', 'Ground Truth'])
                df_pos_cong = pd.DataFrame(columns=['indx', 'patient ID', 'prediction', 'GT', 'logit'])
                df_pos_cong_train = pd.DataFrame(columns=['indx', 'patient ID', 'prediction', 'GT', 'logit'])
                df_pos_cong_name = os.path.join(output_directory, 'positive_cong_test_info' + str(round) + '.xlsx')
                df_pos_cong_name_train = os.path.join(output_directory, 'positive_cong_train_info' + str(round) + '.xlsx')
            else:
                dff_name = os.path.join(output_directory, 'traditional_test_accuracy' + str(round) + '.xlsx')
                dff = pd.DataFrame(columns=['indx', 'Predicted Correct', 'Total Count', 'Ground Truth'])
                df_pos_cong = pd.DataFrame(columns=['indx', 'prediction', 'GT', 'logit'])
                df_pos_cong_train = pd.DataFrame(columns=['indx', 'prediction', 'GT', 'logit'])
                df_pos_cong_name = os.path.join(output_directory, 'positive_cong_test_info' + str(round) + '.xlsx')
                df_pos_cong_name_train = os.path.join(output_directory, 'positive_cong_train_info' + str(round) + '.xlsx')

            # get current training indices
            current_idxs = sampler.idx_current

            # update loaders
            trainer.update_loaders(current_idxs=current_idxs)

            # init epoch and accuracy parameters
            epoch = 0
            acc = 0.0
            unlabeled_epoch = 0

            # reset model
            trainer.reset_model()

            # save checkpoints only in last round
            if round == NUM_ROUND - 1:
                save_checkpoint = True
            else:
                save_checkpoint = False

            # start training
            while acc < args.min_acc:
                # train for this epoch
                if args.eval == 'Patient':
                    acc, preds_and_logits_train, model, opt, input_images, output, preds = trainer.training_participants(epoch, save_checkpoint=save_checkpoint,
                                                                   output_dir=output_directory, round=round)
                else:
                    acc, preds_and_logits_train, model, opt, input_images, output, preds = trainer.training(epoch, save_checkpoint=save_checkpoint,
                                                                                                                         output_dir=output_directory, round=round)
                if args.track_switch_events:
                    switches = trainer.track_unlabeled_forgetting_statistics(unlabeled_epoch)
                else:
                    switches = -1
                unlabeled_epoch += 1

                # track test statistics if specified
                if args.track_test_fevents and epoch % args.eval_test_epoch == 0:
                    print('Testing...')
                    if args.eval == 'Patient':
                        _ = trainer.testing_participants(corrupted=False)
                    else:
                        _ = trainer.testing()

                # increment epoch counter
                epoch += 1

            output_checkpoint = os.path.join(output_directory, "pytorch_ckpt_" + str(round) + ".tar")
            if args.train_type == 'positive_congruent':
                torch.save(
                    {
                        # "model_state_dict": model.state_dict(),
                        # "optimizer_state_dict": opt.state_dict(),
                        "round": round,
                        "input_images": input_images,  # tensor of training images at this round
                        "old_output": output,  # tensor of logits at this round
                        "old_preds": preds  # tensor of preds at this round
                    },
                    output_checkpoint,
                )

            if args.train_type == 'positive_congruent' and round > 0:
                os.remove(os.path.join(base_dir, args.dataset + args.eval, args.architecture,
                                       args.start_strategy, args.out_dir,
                                       "pytorch_ckpt_" + str(round - 1) + ".tar"))

            df_pos_cong_train['indx'] = preds_and_logits_train['indx']
            if args.eval == 'Patient':
                df_pos_cong_train['patient ID'] = preds_and_logits_train['id']
            df_pos_cong_train['prediction'] = preds_and_logits_train['prediction']
            df_pos_cong_train['GT'] = preds_and_logits_train['ground truth']
            df_pos_cong_train['logit'] = preds_and_logits_train['logit']

            # test statistics
            print('Testing normal data....')

            if args.eval == 'Patient':
                accs, preds_and_logits = trainer.testing_participants(corrupted=False, round=round)
            else:
                accs, preds_and_logits = trainer.testing(corrupted=False, round=round)

            if args.dataset == 'CIFAR10' and args.corruption is not None:
                print('Testing corruption type: ' + args.corruption)
                corrupted = trainer.testing(corrupted=True)
                corr_acc = corrupted['test_acc']
            else:
                corr_acc = -1
            test_acc = accs['test_acc']
            test_precision = accs['test_precision']
            test_recall = accs['test_recall']
            ef_acc = accs['ef_acc']
            switch_acc = accs['switch_acc']
            easy_acc = accs['easy_acc']
            difficult_acc = accs['difficult_acc']
            if args.eval == 'Patient':
                patient_id = accs['gt_patient_count'].keys()
                pred_patient_label_count = accs['pred_patient_label_count'].values()
                gt_patient_count = accs['gt_patient_count'].values()
                gt_labels = accs['gt_labels'].values()

            df.loc[round, 'Test Acc'] = test_acc
            df['Test Precision'] = df['Test Precision'].astype('object')
            df['Test Recall'] = df['Test Recall'].astype('object')
            df.at[round, 'Test Precision'] = list(test_precision)
            df.at[round, 'Test Recall'] = list(test_recall)
            df.loc[round, 'EF Test Acc'] = ef_acc
            df.loc[round, 'Switch Test Acc'] = switch_acc
            df.loc[round, 'Easy Test Acc'] = easy_acc
            df.loc[round, 'Difficult Test Acc'] = difficult_acc
            df.loc[round, 'Corrupted Acc'] = corr_acc
            df.loc[round, 'Prediction Switches'] = switches
            df.loc[round, 'Samples'] = len(current_idxs)
            if args.eval == 'Patient':
                dff['ID'] = patient_id
                dff['Predicted Correct'] = pred_patient_label_count
                dff['Total Count'] = gt_patient_count
                dff['Ground Truth'] = gt_labels

                df_pos_cong['indx'] = preds_and_logits['indx']
                df_pos_cong['patient ID'] = preds_and_logits['id']
                df_pos_cong['prediction'] = preds_and_logits['prediction']
                df_pos_cong['GT'] = preds_and_logits['ground truth']
                df_pos_cong['logit'] = preds_and_logits['logit']
            else:
                df_pos_cong['indx'] = preds_and_logits['indx']
                df_pos_cong['prediction'] = preds_and_logits['prediction']
                df_pos_cong['GT'] = preds_and_logits['ground truth']
                df_pos_cong['logit'] = preds_and_logits['logit']

            # query new samples
            if args.strategy == 'mc_update':
                print('calculating mc updates for all unused samples')
                trainer.calculate_mc_updates()
                new_idxs = sampler.query(NUM_QUERY, trainer.sorted_mclist)
            elif args.strategy == 'least_conf' or args.strategy == 'entropy' or args.strategy == 'margin':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'badge':
                print('calculating gradient embeddings')
                embeddings = trainer.get_badge_embeddings()
                new_idxs = sampler.query(NUM_QUERY, embeddings)
            elif args.strategy == 'idealBadge':
                print('Sampling from precalculated embeddings')
                new_idxs = sampler.query(NUM_QUERY)
            elif args.strategy == 'gradcon':
                print('calculating gradcon scores')
                gradcon_scores = trainer.get_gradcon_scores()
                new_idxs = sampler.query(NUM_QUERY, gradcon_scores)
            elif args.strategy == 'reversed_EF_switchsampling' or args.strategy == 'EF_switchsampling':
                # get statistics
                switching_list = trainer.unlabeled_variety
                indices = trainer.unlabeled_seen
                # arange in input structure
                input_structure = {}
                input_structure['switching_frequency'] = switching_list[indices == 1]
                ind_list = np.where(indices == 1)[0]
                input_structure['indices'] = ind_list
                new_idxs = sampler.query(NUM_QUERY, input_structure)
            elif args.strategy == 'patient_diverse':
                input_structure = {}
                input_structure['IDs'] = trainer.unlabeled_loader.dataset.ID
                new_idxs = sampler.query(NUM_QUERY, input_structure)
            elif args.strategy == 'patient_diverse_entropy':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'patient_diverse_entropy_macro':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'patient_diverse_margin':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'patient_diverse_least_conf':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'patient_diverse_badge':
                print('calculating gradient embeddings')
                embeddings = trainer.get_badge_embeddings()
                new_idxs = sampler.query(NUM_QUERY, embeddings)
            elif args.strategy == 'clinically_diverse':
                input_structure = {}
                input_structure['bio'] = trainer.unlabeled_loader.dataset.bio
                new_idxs = sampler.query(NUM_QUERY, input_structure)
            elif args.strategy == 'clinically_diverse_entropy':
                print('calculating probabilities')
                probs = trainer.get_probs()
                new_idxs = sampler.query(NUM_QUERY, probs)
            elif args.strategy == 'clinically_diverse_badge':
                print('calculating gradient embeddings')
                embeddings = trainer.get_badge_embeddings()
                new_idxs = sampler.query(NUM_QUERY, embeddings)
            else:
                new_idxs = sampler.query(NUM_QUERY)

            # update sampler
            sampler.update(new_idx=new_idxs)

            # save forgetting event statistics
            # print('Saving example forgetting statistics')
            # trainer.save_statistics(output_directory, round=round)
            print('clearing all statistics')
            trainer.clear_statistics()
            if args.eval == 'Patient':
                dff.to_excel(dff_name)
                #  to monitor NFR
                df_pos_cong.to_excel(df_pos_cong_name)
                df_pos_cong_train.to_excel(df_pos_cong_name_train)
            else:
                if args.train_type == 'positive_congruent':
                    df_pos_cong.to_excel(df_pos_cong_name)
                    df_pos_cong_train.to_excel(df_pos_cong_name_train)

        # save to dataframe
        df.to_excel(df_name)
        shutil.copy2(df_name, args.out_dir)

    elif args.run_status == 'test':
        # test over all epochs
        trainer.testing()
    else:
        raise (Exception('please set args.run_status=train or test'))
    trainer.writer.close()


# start main
if __name__ == "__main__":
    main()
