import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):

        self.notes=''

        self.vocab_path_train = './data/multi-woz-2.1-processed/vocab'
        self.vocab_path_eval = None
        self.data_path = './data/multi-woz-2.1-processed/'
        self.data_file = 'data_for_damd_fix.json'
        self.dev_list = 'data/MultiWOZ_2.1/valListFile.txt'
        self.test_list = 'data/MultiWOZ_2.1/testListFile.txt'
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.multi_acts_path = 'data/multi-woz-2.1-processed/multi_act_mapping_train.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.fix_data=True

        #key training settings
        self.spv_proportion=50
        self.model_act=True
        self.save_type='max_score'#'min_loss'/'max_reward'
        self.mixed_train=False
        self.dataset=1 # 0 for multiwoz2.0, 1 for multiwoz2.1
        self.example_log=True
        self.delex_as_damd = True 
        self.gen_db=False # critical setting. Only True when we use posterior model to generate belief state and db result.
        self.turn_level=True # turn-level training or session-level training
        self.input_history=False # whether or not add the whole dialog history into the training sequence if train with turn-level 
        self.input_prev_resp=True # whether or not add the prev response into the training sequence if input_history is False
        #pretrain:
        self.posterior_train=False
        #VLtrain:
        self.VL_with_kl=True 
        self.PrioriModel_path='to be generated'
        self.PosteriorModel_path='to be generated'
        self.VL_ablation=False
        #STtrain:
        self.fix_ST=True # whether add straight through trick
        self.ST_resp_only=True #whether calculate cross-entropy on response only
        #evaluation:
        self.fast_validate=True
        self.eval_batch_size=32
        self.gpt_path = '../distilgpt2'
        self.val_set='test'
        self.col_samples=True #collect wrong predictions samples
        self.use_existing_result=True

        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = True
        self.use_true_domain_for_ctr_train = True

        self.post_loss_weight=0.5
        self.kl_loss_weight=0.5
        self.debugging=False
        

        self.loss_reg=True
        self.divided_path='to be generated'
        self.gradient_checkpoint=False
        self.fix_loss=False

        self.sample_type='top1'#'topk'
        self.topk_num=10#only when sample_type=topk
        

        # experiment settings
        self.mode = 'train'
        self.cuda = True
        self.cuda_device = [0]
        self.exp_no = ''
        self.seed = 11
        self.save_log = True # tensorboard 
        self.evaluate_during_training = True # evaluate during training
        self.truncated = False

        # training settings
        self.lr = 1e-4
        self.warmup_steps = -1 
        self.warmup_ratio= 0.2
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 4
        self.batch_size = 8

        self.lr_decay = 0.5
        self.use_scheduler=True
        self.epoch_num = 50
        self.early_stop=False
        self.early_stop_count = 5
        self.weight_decay_count = 10
        
        self.only_target_loss=True # only calculate the loss on target context

        self.clip_grad=True

        # evaluation settings
        self.eval_load_path = 'to be generated'
        self.model_output = 'to be generated'
        self.eval_per_domain = False

        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_db_pointer = False
        self.use_true_prev_resp = False

        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_all_previous_context = True


        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.log_path = ''
        self.low_resource = False


        # model settings
        self.vocab_size = 3000
        self.enable_aspn = True
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False
        #useless settings
        self.multi_acts_training = False
        self.same_eval_as_cambridge = False
        self.same_eval_act_f1_as_hdsa = False

        #parameters for rl training
        self.rl_train=False
        self.delex_resp=True
        self.on_policy=True
        self.fix_DST=False # use a fixed DST model
        self.DST_path=''

        # reward value
        self.resp_punish=-1
        self.rate_th=0.5


        self.entity_provide_reward=1
        self.no_entity_provide_reward=0
        self.no_repeat_ask_reward=1
        self.repeat_ask_reward=-1
        self.no_miss_answer_reward=1
        self.miss_answer_reward=-1
        
        self.non_neg_reward=False # non-negative reward value

        self.rl_dial_per_epoch=512
        self.rl_save_path='RL_exp'
        self.rl_iterate=False # 
        self.rl_iterate_num=1
        # turn level reward: evaluate every turn
        # session level reward: evaluate the whole dialog and punish dialogs that are too long
        self.turn_level_reward=False
        
        self.validate_mode='offline' #offline/online
        self.rl_for_bspn=True # whether calculate the loss on bspn during policy gradient
        self.only_bleu_reward=False
        
        self.init_eval=False # whether evaluate the model before training

        self.test_grad=False

        #user simulator setting
        self.rl_with_us=True # whether or not interact with user simulator 
        self.train_us=False # setting when pre-training user simulator 
        self.train_sys=False
        self.user_nlu=False # add NLU module of user simulator
        self.strict_eval=False
        
        self.joint_train=True # train DS and US together in RL exp
        self.joint_train_us=True
        self.joint_train_ds=True
        self.goal_from_data=True # whether or not use goals in original data
        self.traverse_data=True # traverse all data in training set for one RL epoch
        self.save_by_reward=True # save the model with max average reward

        self.sys_act_ctrl=False
        self.simple_reward=False
        self.simple_training=False
        self.transpose_batch=False
        

        self.DS_path="experiments_21/DS_base/best_score_model"
        self.US_path="experiments_21/US_base/best_loss_model"
        self.DS_device=2
        self.US_device=3
        self.fix_db=True
        self.add_end_reward=False
        self.ctrl_lr=False

        self.interaction_batch_size=32
        self.training_batch_size=8
        self.rl_accumulation_steps=4

        self.test_unseen_act=False
        
        self.eval_resp_prob=False
        self.add_rl_baseline=False
        self.add_resp_reward=False
        self.RL_ablation=False
        self.reward_ablation=False
        self.iterative_update=False

        self.eval_as_simpletod=True
        self.beam_search=False
        self.beam_size_u=3
        self.beam_size_s=8

        self.train_modular=False
        self.modular='dst'
        self.combine_eval=False
        self.gpt_path1=''
        self.gpt_path2=''
        self.gpt_path3=''

        self.jsa_ablation=False
        self.save_prob=False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and mode in ['semi_ST', 'semi_VL', 'semi_jsa', 'train', 'pretrain']:
            if self.dataset==0:
                #file_handler = logging.FileHandler('./log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
                file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
            elif self.dataset==1:
                #file_handler = logging.FileHandler('./log21/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
                file_handler = logging.FileHandler('./log21/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
        elif 'test' in mode and os.path.exists(self.eval_load_path):
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            file_handler.setLevel(logging.INFO)
        else:
            pass
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()