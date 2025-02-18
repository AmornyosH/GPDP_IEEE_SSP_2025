import torch
import numpy as np
import matplotlib.pyplot as plt

gpdp_record = torch.load('GPDP/training_records/GPDP_Walker2d-v5_exact_gp_training_records')
# gpdp_record2 = torch.load('GPDP/training_records/GPDP_Walker2d-v5_exact_gp_training_records_no_alter')
gpdp_loss = gpdp_record['loss_append']
# gpdp_loss2 = gpdp_record2['loss_append']

plt.plot(np.arange(len(gpdp_loss)), gpdp_loss, label='Altered')
# plt.plot(np.arange(len(gpdp_loss2)), gpdp_loss2, label='No Altered')

plt.legend()
plt.xlabel('Gradient step')
plt.ylabel('Negative MLL')
plt.show()