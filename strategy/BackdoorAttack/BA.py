import time
import os
import torch
import random
from data.data_utils import ImageHelper
from strategy.BackdoorAttack.BAServer import BAServer
from strategy.BackdoorAttack.BAClient import BAClient
from configs import args_parser


class BATrainer:
    def __init__(self):
        print("We are conduct federated training using FedAVG algorithm... ")

        self.args = args_parser()

        # create server and clients
        helper = ImageHelper()
        local_dataloaders, local_data_sizes, test_dataloader, test_poison_dataloader = helper.load_data()
        all_clients = [BAClient(_id, local_dataloader, local_data_size)
                       for (_id, local_dataloader), local_data_size
                       in zip(local_dataloaders, local_data_sizes)]
        mal_indices = random.sample(list(range(self.args.n_clients)), int(self.args.n_clients * self.args.mal_ratio))
        if self.args.trigger_type == 'semantic' or self.args.trigger_type == 'swap':
            mal_data_loaders = helper.poison_train_dataset()
            for idx, mal_data_loader in zip(mal_indices, mal_data_loaders):
                all_clients[idx].malicious = True
                all_clients[idx].mal_data_loader = mal_data_loader
        elif self.args.trigger_type == 'pixel' or self.args.trigger_type == 'swap':
            for idx in mal_indices:
                all_clients[idx].malicious = True
        self.server = BAServer(all_clients, test_dataloader, test_poison_dataloader, mal_indices)

        self.current_round = self.server.current_round
        self.results = {'loss': [], 'accuracy': [], 'poison_loss': [], 'poison_accuracy': []}

    def begin_train(self):
        localtime = time.localtime(time.time())
        saved_model_path = os.path.join('../saved_models',
                                        f"{self.args.strategy}_"
                                        f"{localtime[1]:02}{localtime[2]:02}"
                                        f"{localtime[3]:02}{localtime[4]:02}")
        saved_results_path = os.path.join('../ijcai22_results', self.args.strategy)
        start_round = self.current_round
        for _ in range(start_round, self.args.rounds):
            start_time = time.time()

            self.current_round += 1

            # participants selection phase
            self.server.select_participants()

            # training and aggregating phase
            self.server.local_train_aggregate()

            # evaluation phase
            test_acc, test_loss = self.server.validate()
            self.results['accuracy'].append(test_acc)
            self.results['loss'].append(test_loss)
            test_poison_acc, test_poison_loss = self.server.validate_poison()
            self.results['poison_accuracy'].append(test_poison_acc)
            self.results['poison_loss'].append(test_poison_loss)
            print(f"[Round: {self.current_round: 04}], "
                  f"Accuracy on testset: {test_acc: .4f}, "
                  f"Loss on testset: {test_loss: .4f}, "
                  f"Poison accuracy on testset: {test_poison_acc: .4f}, "
                  f"Poison loss on testset: {test_poison_loss: .4f}, "
                  f"Time spent: {time.time() - start_time: .4f} seconds, "
                  f"Estimated time required to complete the training: "
                  f"{(time.time() - start_time) * (self.args.rounds - self.current_round) / 3600}"
                  f" hours.\n")

            # save global model every {global_config['record_step']} rounds
            if self.current_round % self.args.record_step == 0:
                if not os.path.exists(saved_model_path):
                    os.makedirs(saved_model_path)
                torch.save(self.server.global_model.state_dict(),
                           os.path.join(saved_model_path,
                                        f"{self.args.strategy}"
                                        f"_round_{self.current_round}.pth"))

        # save the final results
        if not os.path.exists(saved_results_path):
            os.makedirs(saved_results_path)
        torch.save(self.results, os.path.join(saved_results_path,
                                              f"{self.args.dataset}_"
                                              f"{self.args.model_name}_"
                                              # f"{self.args.strategy}_"
                                              f"{localtime[1]:02}{localtime[2]:02}"
                                              f"{localtime[3]:02}{localtime[4]:02}_"
                                              # f"participatingratio{self.args.participating_ratio}_"
                                              f"malboost{self.args.mal_boost}_"
                                              f"participatingmalratio{self.args.participating_mal_ratio}_"
                                              f"defense{self.args.defense}_"
                                              # f"scale{self.args.scale}_"
                                              f"triggertype{self.args.trigger_type}.pt"))
