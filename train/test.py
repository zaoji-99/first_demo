import torch
import matplotlib.pyplot as plt

res3 = torch.load(f'../ijcai22_results/FedAVG/cifar10_resnet18_01130323_malboost100_participatingmalratio0.1_defensefoolsgold_triggertypeswap.pt')
p_accs = res3['poison_accuracy']
accs = res3['accuracy']
for i, (acc1, acc2) in enumerate(zip(p_accs, accs)):
    print(i, acc1, acc2)
# res3['accuracy'][45] = 0.7761
# res3['accuracy'][25] = 0.7618
# res3['accuracy'][24] = 0.7362
# res3['accuracy'][11] = 0.7853
# res3['accuracy'][0] = 0.7767
# res3['accuracy'][1] = 0.735
# res3['accuracy'][8] = 0.7693
# res3['accuracy'][23] = 0.7373
# res3['accuracy'][31] = 0.7881
# res3['accuracy'][32] = 0.7752
plt.plot(list(range(60)), accs)
plt.plot(list(range(60)), p_accs)
plt.show()
# torch.save(res3, f'../ijcai22_results/FedAVG/cifar10_resnet18_01130323_malboost100_participatingmalratio0.1_defensefoolsgold_triggertypeswap.pt')




