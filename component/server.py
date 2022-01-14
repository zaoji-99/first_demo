import torch
import random
import os
import re
import copy
import torch.optim as optim
from models.create_model import create_model
from component.evaluation import evaluate_accuracy
from configs import args_parser


class Server:
    def __init__(self, clients, test_loader):
        self.args = args_parser()
        self.global_model = create_model(self.args.dataset, self.args.model_name)
        print(f"We are running {self.args.model_name} on {self.args.dataset}\n")
        print(self.global_model)
        self.args.resumed = True if self.args.resumed == 'True' else False
        if self.args.resumed:
            self.global_model.load_state_dict(torch.load(
                os.path.join('../saved_models', self.args.resumed_name)))
        self.clients = clients
        self.test_loader = test_loader
        self.total_size = 0
        self.participants = None
        self.current_round = 0
        if self.args.resumed:
            self.current_round = int(re.findall(r'\d+\d*', self.args.resumed_name.split('/')[1])[0])

    # should be overridden in some subclass
    def select_participants(self):
        self.current_round += 1
        self.total_size = 0
        to_be_selected = list(range(len(self.clients)))
        self.participants = random.sample(to_be_selected,
                                          round(self.args.participating_ratio * len(self.clients)))
        for client_id in self.participants:
            self.total_size += self.clients[client_id].local_data_size
        print(f'Participants in round {self.current_round}: {[client_id for client_id in self.participants]}, '
              f'Total size in round {self.current_round}: {self.total_size}')

    # should be overridden in some subclass
    def local_train_aggregate(self):
        aggregated_global_model = self.global_model.state_dict()
        for client_id in self.participants:
            # Information for client
            local_dataloader = self.clients[client_id].local_dataloader
            local_data_size = self.clients[client_id].local_data_size
            local_model = copy.deepcopy(self.global_model)
            local_model.load_state_dict(self.global_model.state_dict())
            local_optimizer = optim.SGD(local_model.parameters(),
                                        lr=self.args.lr,
                                        weight_decay=self.args.weight_decay,
                                        momentum=self.args.momentum)
            local_loss = torch.nn.CrossEntropyLoss()

            # Local training process
            for _ in range(self.args.epochs):
                for X, y in local_dataloader:
                    X = X.to(self.args.device)
                    y = y.to(self.args.device)
                    local_optimizer.zero_grad()
                    train_l = local_loss(local_model(X), y)
                    train_l.backward()
                    local_optimizer.step()

            # Aggregate
            for name, param in local_model.state_dict().items():
                aggregated_global_model[name] = \
                    aggregated_global_model[name] - (self.args.eta / self.args.n_clients) * \
                    (aggregated_global_model[name] - param.data)

        # Load the parameters after training
        self.global_model.load_state_dict(aggregated_global_model)

    def validate(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_accuracy(self.global_model, self.test_loader)
        return test_acc, test_l
