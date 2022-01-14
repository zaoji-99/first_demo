from component.server import Server
from configs import args_parser
from data.data_utils import ImageHelper
from component.evaluation import evaluate_poison_accuracy, evaluate_accuracy
from scipy.linalg import eigh as largest_eigh
from math import pi
import random
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn.metrics.pairwise as smp


class BAServer(Server):
    def __init__(self, clients, test_loader, test_poison_dataloader, mal_indices):
        super(BAServer, self).__init__(clients, test_loader)
        self.test_poison_dataloader = test_poison_dataloader
        self.mal_indices = mal_indices
        self.benign_indices = list(set(list(range(self.args.n_clients))) - set(self.mal_indices))
        self.args = args_parser()
        self.helper = ImageHelper()
        # poison_rounds = [int(i) for i in self.args.poison_rounds.split(',')]
        # self.args.poison_rounds = poison_rounds
        poison_rounds = np.arange(self.current_round + 1,
                                  self.current_round + self.args.poison_rounds + 1)
        self.args.continuity = True if self.args.continuity == 'True' else False
        if self.args.continuity:
            self.args.poison_rounds = poison_rounds.tolist()
        else:
            np.random.seed(30)
            whether_poison = np.random.uniform(0, 1, self.args.poison_rounds) >= (1 - self.args.poison_prob)
            self.args.poison_rounds = set((poison_rounds * whether_poison).tolist())
            self.args.poison_rounds.remove(0)
        self.participant_nums = round(self.args.n_clients * self.args.participating_ratio)
        self.benign_nums = 0
        print(f'\nparameter settings:\n'
              f'  --dataset: {self.args.dataset}\n'
              f'  --model: {self.args.model_name}\n'
              f'  --resumed: {self.args.resumed}\n'
              f'  --rounds: {self.args.rounds}\n'
              f'  --trigger_type: {self.args.trigger_type}\n'
              f'  --eta: {self.args.eta}\n'
              f'  --batch_size: {self.args.batch_size}\n'
              f'  --epochs: {self.args.epochs}\n'
              f'  --lr: {self.args.lr}\n'
              f'  --weight_decay: {self.args.weight_decay}\n'
              f'  --num_mal_samples: {self.args.num_mal_samples}\n'
              f'  --participating_mal_ratio: {self.args.participating_mal_ratio}\n'
              f'  --poison_rounds: {self.args.poison_rounds}\n'
              f'  --mal_epochs: {self.args.mal_epochs}\n'
              f'  --mal_lr: {self.args.mal_lr}\n'
              f'  --mal_weight_decay: {self.args.mal_weight_decay}\n'
              f'  --mal_boost: {self.args.mal_boost}\n'
              f'  --alpha: {self.args.alpha}\n'
              f'  --scale: {self.args.scale}\n'
              f'  --defense: {self.args.defense}\n'
              f'  --malicious clients: {self.mal_indices}\n'
              f'  --benign clients: {self.benign_indices}\n')

    def select_participants(self):
        self.current_round += 1
        self.total_size = 0
        if len(self.args.poison_rounds) == 0 or self.current_round not in self.args.poison_rounds:
            self.participants = random.sample(list(range(self.args.n_clients)), self.participant_nums)
            self.benign_nums = self.participant_nums
        else:
            if self.current_round in self.args.poison_rounds:
                mal_nums = int(self.participant_nums * self.args.participating_mal_ratio)
                self.benign_nums = self.participant_nums - mal_nums
                self.participants = []
                self.participants = self.participants + random.sample(self.mal_indices, mal_nums)
                self.participants = self.participants + random.sample(self.benign_indices, self.benign_nums)
        for client_id in self.participants:
            if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                if self.args.trigger_type == 'semantic' or self.args.trigger_type == 'swap':
                    self.total_size += self.args.batch_size * self.args.batches_of_mal_data
            else:
                self.total_size += self.clients[client_id].local_data_size
        print(f'Participants in round {self.current_round}: {[client_id for client_id in self.participants]}, '
              f'Total size in round {self.current_round}: {self.total_size}, '
              f'Benign participants: {self.benign_nums}')

    def local_train_aggregate(self):
        K = self.participant_nums
        mal_nums = K - self.benign_nums
        model_updates = {}
        if self.args.defense == 'ours':
            for client_id in self.participants:
                local_dataloader = self.clients[client_id].local_dataloader
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(self.global_model.state_dict())
                if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                    if self.args.trigger_type == 'semantic' or self.args.trigger_type == 'swap':
                        local_dataloader = self.clients[client_id].mal_data_loader
                    lr = self.args.mal_lr
                    weight_decay = self.args.mal_weight_decay
                    momentum = self.args.mal_momentum
                    epochs = self.args.mal_epochs
                else:
                    lr = self.args.lr
                    weight_decay = self.args.weight_decay
                    momentum = self.args.momentum
                    epochs = self.args.epochs
                local_optimizer = optim.SGD(local_model.parameters(),
                                            lr=lr, weight_decay=weight_decay, momentum=momentum)
                local_loss = torch.nn.CrossEntropyLoss()

                # Local training process
                local_model = self.local_train(client_id, local_dataloader, local_model, local_loss, epochs, local_optimizer)

                # local test and local test poison
                if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                    test_local_acc, test_local_l = evaluate_accuracy(local_model, local_dataloader)
                    test_local_acc_poison, test_local_l_poison = evaluate_poison_accuracy(
                        local_model, self.test_poison_dataloader, self.helper)
                    print(f"[Attack Round: {self.current_round: 04}], "
                          f"Malicious id: {client_id}, "
                          f"Local accuracy on local data: {test_local_acc: .4f}, "
                          f"Local loss on local data: {test_local_l: .4f} "
                          f"Local poison accuracy on testset: {test_local_acc_poison: .4f}, "
                          f"Local poison loss on testset: {test_local_l_poison: .4f} ")

                # Calculate local model update
                for (name, param), (_, local_param) in zip(
                        self.global_model.state_dict().items(), local_model.state_dict().items()):
                    local_update = (param.data - local_param.data).view(1, -1)
                    if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                        if self.args.mal_boost > 1:
                            local_update = self.args.mal_boost * local_update / mal_nums
                    if name not in model_updates:
                        model_updates[name] = local_update
                    else:
                        model_updates[name] = torch.cat((model_updates[name], local_update), dim=0)

            # detect malicious
            cos = torch.nn.CosineSimilarity(dim=0)
            abnormal_idxs = set()
            cos_sims = {}
            if K - self.benign_nums == 1:
                pass
            elif self.current_round in self.args.poison_rounds:
                if self.args.model_name == 'resnet18':
                    last_layer_updates = model_updates['linear.weight']
                else:
                    last_layer_updates = model_updates['fc2.weight']
                # wv = self.foolsgold(last_layer_updates.cpu().numpy())
                # for i, j in enumerate(wv):
                #     if j == 0:
                #         abnormal_idxs.add(i)
                for i in range(K - 1):
                    for j in range(i + 1, K):
                        cos_sim = cos(last_layer_updates[j].view(-1, 1), last_layer_updates[i].view(-1, 1)).item()
                        # print(f'cosine similarity between {i} and {j}: {cos_sim}')
                        if i in cos_sims:
                            cos_sims[i].append(cos_sim)
                        else:
                            cos_sims[i] = [cos_sim]
                sort_cos_sims = []
                for cos_sim in cos_sims.values():
                    sort_cos_sims.extend(cos_sim)
                sort_cos_sims.sort(reverse=True)
                i = 0
                while len(abnormal_idxs) < K - self.benign_nums:
                    max_cos_sim = sort_cos_sims[i]
                    for k, cs in cos_sims.items():
                        try:
                            idx = cs.index(max_cos_sim)
                            print(max_cos_sim, k, k + 1 + idx)
                            abnormal_idxs.add(k)
                            abnormal_idxs.add(k + 1 + idx)
                            i += 1
                            break
                        except ValueError:
                            continue
                print(f'possibly malicious client: {[self.participants[idx] for idx in abnormal_idxs]}')

            # Calculate global model update
            global_update = {}
            for name, layer_updates in model_updates.items():
                if 'num_batches_tracked' in name:
                    global_update[name] = torch.sum(model_updates[name][K - self.benign_nums:]) / self.benign_nums
                    continue
                layer_updates_copy = F.normalize(layer_updates)
                local_norms = []
                for i in range(K):
                    local_norms.append(torch.norm(layer_updates[i]))

                if abnormal_idxs:
                    abnormal_idxs_copy = list(set(abnormal_idxs))
                    abnormal_idxs_copy.sort()
                    first_idx = abnormal_idxs_copy[0]
                    for idx in abnormal_idxs_copy[1:]:
                        layer_updates[first_idx] = layer_updates[first_idx] + layer_updates[idx]
                    layer_updates[first_idx] = layer_updates[first_idx] / len(abnormal_idxs_copy)
                    local_norms[first_idx] = torch.norm(layer_updates[first_idx])
                    layer_updates_copy[first_idx] = layer_updates[first_idx] / local_norms[first_idx]
                    remove_idxs = [idx - i for i, idx in enumerate(abnormal_idxs_copy[1:])]
                    for idx in remove_idxs:
                        layer_updates_copy = layer_updates_copy[torch.arange(layer_updates_copy.size(0)) != idx]
                        local_norms.pop(idx)

                    # perturbation
                    if K - self.benign_nums == 1:
                        pass
                    elif self.current_round in self.args.poison_rounds:
                        theta = torch.tensor(np.random.uniform(2 * pi / 5, pi / 2)).to(self.args.device)
                        direction = torch.normal(0, 1, layer_updates_copy[first_idx].size()).to(self.args.device)
                        direction /= torch.norm(direction)
                        layer_updates_copy[first_idx] += torch.norm(layer_updates_copy[first_idx]) * torch.tan(theta) * direction
                        layer_updates_copy[first_idx] /= torch.norm(layer_updates_copy[first_idx])
                else:
                    layer_updates_copy = F.normalize(layer_updates)

                N = layer_updates_copy.size(0)
                X = torch.matmul(layer_updates_copy, layer_updates_copy.T)
                evals_large, evecs_large = largest_eigh(X.detach().cpu().numpy(), eigvals=(N - N, N - 1))
                evals_large = torch.tensor(evals_large)[-1].to(self.args.device)
                evecs_large = torch.tensor(evecs_large)[:, -1].to(self.args.device)
                principal_layer_update = torch.matmul(evecs_large.view(1, -1), layer_updates_copy).T \
                                         / torch.sqrt(evals_large)
                principal_layer_update = principal_layer_update / torch.norm(principal_layer_update)
                positive = 0
                for i in range(N):
                    if cos(principal_layer_update, layer_updates_copy[i].view(-1, 1)).item() > 0:
                        positive += 1
                if positive < N - positive:
                    principal_layer_update = -principal_layer_update
                variance = 0
                min_norm = min(local_norms)
                for i in range(N):
                    cos_sim = cos(principal_layer_update.view(-1, 1), layer_updates_copy[i].view(-1, 1)).item()
                    variance += min_norm * cos_sim
                variance /= N
                variance *= self.args.scale
                global_update[name] = principal_layer_update * variance

            # Update the global model
            updated_global_model = self.global_model.state_dict()
            for name, param in updated_global_model.items():
                updated_global_model[name] = param.data - global_update[name].view(param.size())
            self.global_model.load_state_dict(updated_global_model)
        else:
            # aggregated_global_model = self.global_model.state_dict()
            # local_updates = []
            """
            mal_updates = None
            mal_indices = []
            """
            for client_id in self.participants:
                # local_data_size = 0
                # if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                #     if self.args.trigger_type == 'semantic':
                #         local_data_size = self.args.batch_size * self.args.batches_of_mal_data
                # else:
                #     local_data_size = self.clients[client_id].local_data_size
                local_dataloader = self.clients[client_id].local_dataloader
                local_model = copy.deepcopy(self.global_model)
                # local_model_benign = copy.deepcopy(self.global_model)
                local_model.load_state_dict(self.global_model.state_dict())
                if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                    if self.args.trigger_type == 'semantic' or self.args.trigger_type == 'swap':
                        local_dataloader = self.clients[client_id].mal_data_loader
                    lr = self.args.mal_lr
                    weight_decay = self.args.mal_weight_decay
                    momentum = self.args.mal_momentum
                    epochs = self.args.mal_epochs
                else:
                    lr = self.args.lr
                    weight_decay = self.args.weight_decay
                    momentum = self.args.momentum
                    epochs = self.args.epochs
                local_optimizer = optim.SGD(local_model.parameters(),
                                            lr=lr, weight_decay=weight_decay, momentum=momentum)
                local_loss = torch.nn.CrossEntropyLoss()

                # Local training process
                local_model = self.local_train(client_id, local_dataloader, local_model,
                                               local_loss, epochs, local_optimizer)
                # local_updates.append(torch.nn.utils.parameters_to_vector(self.global_model.parameters()) -
                #                      torch.nn.utils.parameters_to_vector(local_model.parameters()))
                # if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                #     local_model_benign = self.local_train(client_id, self.clients[client_id].local_dataloader,
                #                                           local_model_benign, local_loss, epochs, local_optimizer)
                #     cos = torch.nn.CosineSimilarity(dim=1)
                #     for (name, local_param), (_, local_param_begin), (_, global_param) in zip(
                #             local_model.state_dict().items(),
                #             local_model_benign.state_dict().items(),
                #             self.global_model.state_dict().items()):
                #         if 'bn' not in name and 'num_batches_tracked' not in name:
                #             local_update = (global_param.data - local_param.data).view(1, -1)
                #             local_update_begin = (global_param.data - local_param_begin.data).view(1, -1)
                #             print(client_id, name, cos(local_update, local_update_begin))

                # local test poison
                if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                    """
                    mal_indices.append(client_id)
                    mal_update = torch.nn.utils.parameters_to_vector(self.global_model.parameters()) - \
                                 torch.nn.utils.parameters_to_vector(local_model.parameters())
                    if mal_updates is None:
                        mal_updates = mal_update.view(1, -1)
                    else:
                        mal_updates = torch.cat((mal_updates, mal_update.view(1, -1)))
                    """
                    test_local_acc, test_local_l = evaluate_accuracy(local_model, local_dataloader)
                    test_local_acc_poison, test_local_l_poison = evaluate_poison_accuracy(
                        local_model, self.test_poison_dataloader, self.helper)
                    print(f"[Attack Round: {self.current_round: 04}], "
                          f"Malicious id: {client_id}, "
                          f"Local accuracy on local data: {test_local_acc: .4f}, "
                          f"Local loss on local data: {test_local_l: .4f} "
                          f"Local poison accuracy on testset: {test_local_acc_poison: .4f}, "
                          f"Local poison loss on testset: {test_local_l_poison: .4f} ")

                for (name, param), (_, local_param) in zip(
                        self.global_model.state_dict().items(), local_model.state_dict().items()):
                    local_update = (param.data - local_param.data).view(1, -1)
                    if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                        if self.args.mal_boost > 1:
                            local_update = self.args.mal_boost * local_update / mal_nums
                    if name not in model_updates:
                        model_updates[name] = local_update
                    else:
                        model_updates[name] = torch.cat((model_updates[name], local_update), dim=0)

                # for (name, param), (_, local_param) in zip(
                #         self.global_model.state_dict().items(), local_model.state_dict().items()):
                #     local_update = (param.data - local_param.data)
                #     if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                #         if self.args.mal_boost > 1:
                #             local_update = self.args.mal_boost * local_update / mal_nums
                #     if name not in global_update:
                #         global_update[name] = local_update
                #     else:
                #         global_update[name] = global_update[name] + local_update

            # Update the global model
            updated_global_model = self.global_model.state_dict()
            # foolsgold
            if self.args.defense == 'foolsgold':
                if self.args.model_name == 'resnet18':
                    last_layer_updates = model_updates['linear.weight']
                else:
                    last_layer_updates = model_updates['fc2.weight']
                wv = self.foolsgold(last_layer_updates.cpu().numpy())
                for name, param in updated_global_model.items():
                    tmp = None
                    for i, j in enumerate(range(len(wv))):
                        if i == 0:
                            tmp = model_updates[name][j] * wv[j]
                        else:
                            tmp += model_updates[name][j] * wv[j]
                    updated_global_model[name] = param.data - 1 / len(wv) * tmp.view(param.size())
            elif self.args.defense == 'diff_privacy':
                for name, param in updated_global_model.items():
                    shape = model_updates[name].size()
                    noise = torch.normal(0, self.args.sigma, shape).to(self.args.device)
                    updated_global_model[name] = param.data - \
                                                 (self.args.eta / self.args.n_clients) * \
                                                 (model_updates[name] + noise).sum(dim=0, keepdim=True).view(param.size())
            else:
                for name, param in updated_global_model.items():
                    updated_global_model[name] = param.data - \
                                                 (self.args.eta / self.args.n_clients) * \
                                                 model_updates[name].sum(dim=0, keepdim=True).view(param.size())
            self.global_model.load_state_dict(updated_global_model)

            # cos similarity among clients
            # cos = torch.nn.CosineSimilarity(dim=0)
            # local_update_num = len(local_updates)
            # for i in range(local_update_num - 1):
            #     for j in range(i + 1, local_update_num):
            #         cos_sim = cos(local_updates[i], local_updates[j])
            #         print(f'Cos similarity between {i} and {j}: {cos_sim}')

    def model_cos_sim(self, global_model, updated_local_model):
        cos_list = list()
        cos_loss = torch.nn.CosineSimilarity(dim=1)
        for (name, global_param), (_, updated_local_param) in zip(
                global_model.state_dict().items(), updated_local_model.state_dict().items()):
            if 'num_batches_tracked' in name:
                continue
            cos = cos_loss(global_param.data.view(1, -1), updated_local_param.data.view(1, -1))
            cos_list.append(cos)
        cos_los_submit = (1 - sum(cos_list) / len(cos_list))
        return cos_los_submit

    def local_train(self, client_id, local_dataloader, local_model, local_loss, epochs, local_optimizer):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            local_optimizer, milestones=[0.2 * self.args.mal_epochs, 0.8 * self.args.mal_epochs], gamma=0.1)
        for _ in range(epochs):
            for batch_idx, batch in enumerate(local_dataloader):
                if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                    X, y, _ = self.helper.get_poison_batch(batch)
                else:
                    X, y = batch
                X = X.to(self.args.device)
                y = y.to(self.args.device)
                local_optimizer.zero_grad()
                train_l = local_loss(local_model(X), y)
                if self.args.alpha < 1:
                    # if self.current_round not in self.args.poison_rounds or not self.clients[client_id].malicious:
                    train_l = self.model_cos_sim(self.global_model, local_model) * (1 - self.args.alpha) \
                          + train_l * self.args.alpha
                train_l.backward()
                local_optimizer.step()
            if self.current_round in self.args.poison_rounds and self.clients[client_id].malicious:
                scheduler.step()
        return local_model

    def foolsgold(self, grads):
        n_clients = len(grads)
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        return wv

    def validate_poison(self):
        with torch.no_grad():
            test_acc, test_l = evaluate_poison_accuracy(self.global_model, self.test_poison_dataloader, self.helper)
        return test_acc, test_l
