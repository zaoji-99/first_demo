import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from configs import args_parser

args = args_parser()

saved_plot_path = os.path.join('../saved_plots',
                               f"{args.strategy}")
if not os.path.exists(saved_plot_path):
    os.makedirs(saved_plot_path)


class Animator:
    def __init__(self, x_axis, y_axes, x_label=None, y_label=None, x_lim=None, y_lim=None, segments=None, legends=None, fmts=None):
        assert len(legends) == len(fmts)
        self.x_label = x_label
        self.y_label = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.segments = segments
        self.legends = legends
        self.fmts = fmts
        self.x_axis = x_axis
        self.y_axes = [[] for _ in range(len(y_axes))]
        for i, y_axis in enumerate(y_axes):
            self.y_axes[i] = y_axis

    def display(self, fname=None, legend_fontsize=24, legend_set=False, legend_pos=None, bbox_to_anchor=None, ncol=1,
                show=False, marker=False, marker_points=None, marker_fmts=None):
        if show:
            params = {
                'font.family': 'serif',
                'font.serif': 'Times New Roman',
                'font.size': 24
            }
            matplotlib.rcParams.update(params)
            # plt.figure(figsize=(7, 7))
            _, ax = plt.subplots(1, 1, figsize=(7, 7))
            for y_axis, legend, fmt in zip(self.y_axes, self.legends, self.fmts):
                ax.plot(self.x_axis, y_axis, fmt, label=legend)
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            if legend_set:
                ax.legend(loc=legend_pos, fontsize=legend_fontsize, bbox_to_anchor=bbox_to_anchor, ncol=ncol, borderaxespad=0)

            ax.grid()
            if self.y_lim:
                ax.set_yticks(np.linspace(self.y_lim[0], self.y_lim[1], self.segments))
            if self.x_lim:
                ax.set_xticks(np.linspace(self.x_lim[0], self.x_lim[1], self.segments))
            if marker:
                if marker_points:
                    for makers, fmt in zip(marker_points, marker_fmts):
                        xs, ys = makers
                        for x, y in zip(xs, ys):
                            ax.scatter(x-1, y, facecolors='none', marker='o', edgecolors=fmt, s=80)
            # plt.savefig(os.path.join(saved_plot_path, fname))
            plt.show()


np.random.seed(30)
poison_rounds = np.arange(1, args.poison_rounds + 1)
whether_poison = np.random.uniform(0, 1, args.poison_rounds) >= (1 - args.poison_prob)
poison_rounds = set((poison_rounds * whether_poison).tolist())
poison_rounds.remove(0)
x_marker = list(poison_rounds)


def main_accuracy():
    """Figure 1: Accuracy of main task"""
    x = list(range(int(args.rounds) + 1))
    step = 5
    x = [x[i] for i in range(10, args.rounds + 1, step)]
    main_task_avg = torch.load(f'../saved_results/FedAVG/cifar10_resnet18_FedAVG_12211130_participatingmalratio0.0_defenseours.pt')
    main_task_ours = torch.load(f'../saved_results/FedAVG/cifar10_resnet18_FedAVG_12211000_participatingmalratio0.0_defenseno.pt')
    main_acc_raw_ours = main_task_avg['accuracy']
    main_acc_ours = [main_acc_raw_ours[i] for i in range(9, args.rounds, step)]
    main_acc_raw_no = main_task_ours['accuracy']
    main_acc_no = [main_acc_raw_no[i] for i in range(9, args.rounds, step)]
    fmt1 = 'r-'
    fmt2 = 'g-'
    legend1 = 'main accuracy - ours'
    legend2 = 'main accuracy - no'
    x_label = 'Communication Rounds'
    y_label = 'Accuracy'
    animator = Animator(x_axis=x, y_axes=[main_acc_ours, main_acc_no], x_label=x_label, y_label=y_label,
                        legends=[legend1, legend2], fmts=[fmt1, fmt2])
    print(sum(main_acc_ours[-10:]) / 10)
    print(sum(main_acc_no[-10:]) / 10)
    print('Difference:', (sum(main_acc_ours[-10:]) - sum(main_acc_no[-10:])) / 10)
    animator.display(fname='main_task.pdf', show=True)


def ba():
    """Figure 2: BA"""
    no_pixel = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112017_malboost100_participatingmalratio0.5_defenseno_triggertypepixel.pt')['poison_accuracy']
    defense_pixel = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112018_malboost100_participatingmalratio0.5_defenseours_triggertypepixel.pt')['poison_accuracy']
    no_semantic = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112019_malboost100_participatingmalratio0.5_defenseno_triggertypesemantic.pt')['poison_accuracy']
    defense_semantic = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112020_malboost100_participatingmalratio0.5_defenseours_triggertypesemantic.pt')['poison_accuracy']
    x = list(range(int(args.poison_rounds)))
    marker_points_no_pixel = (x_marker, [no_pixel[i-1] for i in x_marker])
    marker_points_defense_pixel = (x_marker, [defense_pixel[i-1] for i in x_marker])
    marker_points_no_semantic = (x_marker, [no_semantic[i-1] for i in x_marker])
    marker_points_defense_semantic = (x_marker, [defense_semantic[i-1] for i in x_marker])
    marker_points = [marker_points_no_pixel, marker_points_no_semantic,
                     marker_points_defense_pixel, marker_points_defense_semantic]
    marker_fmts = ['r', 'g', 'r', 'g']
    fmt1 = 'r-'
    fmt2 = 'r--'
    fmt3 = 'g-'
    fmt4 = 'g--'
    legend1 = 'pixel - no'
    legend2 = 'pixel - ours'
    legend3 = 'semantic - no'
    legend4 = 'semantic - ours'
    x_label = 'Communication rounds'
    y_label = 'Backdoor accuracy'
    animator = Animator(x_axis=x, y_axes=[no_pixel, defense_pixel, no_semantic, defense_semantic],
                        x_label=x_label, y_label=y_label,
                        legends=[legend1, legend2, legend3, legend4], fmts=[fmt1, fmt2, fmt3, fmt4])
    animator.display(fname='ba.pdf', legend_set=True, legend_pos=4, bbox_to_anchor=(1, 0.08), show=True,
                     marker=True, marker_points=marker_points, marker_fmts=marker_fmts)


def swap():
    """SWAP"""
    no_swap_res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112315_malboost100_participatingmalratio0.5_defenseno_triggertypeswap.pt')
    no_swap_main_res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112341_malboost100_participatingmalratio0.5_defenseno_triggertypeswap.pt')
    defense_swap_res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01112316_malboost100_participatingmalratio0.5_defenseours_triggertypeswap.pt')
    no_swap_poison = no_swap_res['poison_accuracy']
    no_swap_other = no_swap_res['accuracy']
    no_swap_main = no_swap_main_res['accuracy']
    defense_swap_poison = defense_swap_res['poison_accuracy']
    defense_swap_main = defense_swap_res['accuracy']
    x = list(range(int(args.poison_rounds)))
    marker_points_no_swap = (x_marker, [no_swap_poison[i-1] for i in x_marker])
    marker_points_defense_swap = (x_marker, [defense_swap_poison[i-1] for i in x_marker])
    marker_points = [marker_points_no_swap, marker_points_defense_swap]
    marker_fmts = ['r', 'g']
    fmt1 = 'r-'
    fmt2 = 'r--'
    fmt3 = 'b--'
    fmt4 = 'g-'
    fmt5 = 'g--'
    legend1 = 'flip - no'
    legend2 = 'flip - no, other'
    legend3 = 'flip - no, main'
    legend4 = 'flip - ours'
    legend5 = 'flip - ours, main'
    x_label = 'Communication rounds'
    y_label = 'Accuracy'
    animator = Animator(x_axis=x, y_axes=[no_swap_poison, no_swap_other, no_swap_main, defense_swap_poison, defense_swap_main],
                        x_label=x_label, y_label=y_label,
                        legends=[legend1, legend2, legend3, legend4, legend5], fmts=[fmt1, fmt2, fmt3, fmt4, fmt5])
    animator.display(fname='swap.pdf', show=True, legend_set=True, legend_pos=3, bbox_to_anchor=(0.9, 1.01), ncol=1,
                     marker=True, marker_points=marker_points, marker_fmts=marker_fmts)


def mal_ratio_1():
    defense_semantic_06res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01120042_participatingratio0.1_participatingmalratio0.6_defenseours_triggertypesemantic.pt')
    defense_semantic_07res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01120042_participatingratio0.1_participatingmalratio0.7_defenseours_triggertypesemantic.pt')
    defense_semantic_08res = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01120033_participatingratio0.1_participatingmalratio0.8_defenseours_triggertypesemantic.pt')
    defense_semantic_06poison = defense_semantic_06res['poison_accuracy']
    defense_semantic_06main = defense_semantic_06res['accuracy']
    defense_semantic_07poison = defense_semantic_07res['poison_accuracy']
    defense_semantic_07main = defense_semantic_07res['accuracy']
    defense_semantic_08poison = defense_semantic_08res['poison_accuracy']
    defense_semantic_08main = defense_semantic_08res['accuracy']
    x = list(range(int(args.poison_rounds)))
    marker_points_06poison = (x_marker, [defense_semantic_06poison[i-1] for i in x_marker])
    marker_points_07poison = (x_marker, [defense_semantic_07poison[i-1] for i in x_marker])
    marker_points_08poison = (x_marker, [defense_semantic_08poison[i-1] for i in x_marker])
    marker_points = [marker_points_06poison, marker_points_07poison, marker_points_08poison]
    marker_fmts = ['r', 'g', 'b']
    fmt1 = 'r-'
    fmt2 = 'r--'
    fmt3 = 'g-'
    fmt4 = 'g--'
    fmt5 = 'b-'
    fmt6 = 'b--'
    legend1 = '0.6, B'
    legend2 = '0.6, M'
    legend3 = '0.7, B'
    legend4 = '0.7, M'
    legend5 = '0.8, B'
    legend6 = '0.8, M'
    x_label = 'Communication rounds'
    y_label = 'Accuracy'
    animator = Animator(x_axis=x, y_axes=[defense_semantic_06poison, defense_semantic_06main,
                                          defense_semantic_07poison, defense_semantic_07main,
                                          defense_semantic_08poison, defense_semantic_08main],
                        x_label=x_label, y_label=y_label,
                        legends=[legend1, legend2, legend3, legend4, legend5, legend6], fmts=[fmt1, fmt2, fmt3, fmt4, fmt5, fmt6])
    animator.display(fname='mal_ratio_1.pdf', show=True, legend_set=True,
                     legend_fontsize=18, bbox_to_anchor=(0.01, 1.01), legend_pos=3, ncol=3)


main_accuracy()