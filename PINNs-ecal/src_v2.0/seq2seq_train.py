import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from util import *

class Seq2Seq_Train:
    def __init__(self,
                encoder,
                decoder,
                input_tensor,
                target_tensor,
                n_epochs,
                target_len,
                batch_size,
                learning_rate=0.01,
                opt_alg='adam',
                print_step=1,
                strategy = 'recursive',
                teacher_forcing_ratio=0.5,
                device='cpu',
                loss_figure_name='loss.png'):

        self.encoder = encoder
        self.decoder = decoder
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.n_epochs = n_epochs
        self.target_len = target_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_alg = opt_alg
        self.print_step = print_step
        self.strategy = strategy
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.loss_figure_name = loss_figure_name
        
        # — physics‑informed parameters (learnable) —
        self.k_dmg       = torch.tensor(0.001, requires_grad=True, device=self.device)
        self.k_rec       = torch.tensor(0.0001, requires_grad=True, device=self.device)
        # C_inf: steady-state calibration
        self.C_inf       = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.lambda_pinn = 100

    def train(self,S_cal):
        print('>>> Start training... (be patient: training time varies)')
        # if self.strategy == 'recursive':
        #     self.train_model_recursive()

        if self.strategy == 'teacher_forcing':
            self.train_model_teacher_forcing(S_cal)

        # elif self.strategy == 'mixed':
        #     self.train_model_mixed()

        else:
            assert False, "Please select one of them---[recursive, teacher_forcing, mixed]!"

        print('>>> Finish training!')

    def train_model_teacher_forcing(self,S_cal):
        ### move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        ### get the learnable parameters
        params = []
        params += [x for x in self.encoder.parameters()]
        params += [x for x in self.decoder.parameters()]
        params += [self.k_dmg,self.k_rec,self.C_inf]
        ### define optimizer method
        if self.opt_alg.upper() == 'ADAM':
            optimizer = optim.Adam(params=params, lr=self.learning_rate)
        elif self.opt_alg.upper() == 'SGD':
            optimizer = optim.SGD(params=params, lr=self.learning_rate)
        else:
            assert False, 'This version only supports ADAM and SGD!'

        ### define loss function
        criterion = nn.MSELoss()

        ### calculate number of batch iterations
        n_batches = int(self.input_tensor.shape[1] / self.batch_size)

        ### save loss
        losses = []
        print("PINNS IN ACTION : ")
        for epoch in range(self.n_epochs):
            # print("======== epoch {} out of {} epochs ========".format(epoch,self.n_epochs))
            self.encoder.train()
            self.decoder.train()
            batch_loss = []

            for batch in range(n_batches):
                # print("batch {} out of {} batches".format(batch,n_batches))
                # select data
                #remove unneeded unscaled features
                input_batch = self.input_tensor[:, batch: batch + self.batch_size, :3]

                # target_batch_input means the "luminosity delta", which is given to us
                # we will combine this information with "calibration" as input to decoder each time
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 0:1]
                target_batch_input2 = self.target_tensor[:, batch: batch + self.batch_size, 2:3]
                # target_batch means the "calibration", which is the value we want to predict
                # so target_batch is the real target
                target_batch = self.target_tensor[:, batch: batch + self.batch_size, 1:2]

                # move data to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                target_batch_input = target_batch_input.to(self.device)
                target_batch_input2 = target_batch_input2.to(self.device)
                # outputs tensor
                outputs = torch.zeros(self.target_len, self.batch_size, target_batch.shape[2])

                # initialize hidden state
                # encoder_hidden = self.encoder.init_hidden(batch_size)

                # zero the gradient
                optimizer.zero_grad()
                
                
                # encoder outputs
                encoder_output, encoder_hidden = self.encoder(input_batch)

                # decoder with teacher forcing
                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                # different training strategies
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = target_batch[t, :, :]
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        deltat_feature = target_batch_input2[t,:,:]
                        decoder_input = torch.cat((lumi_feature, deltat_feature, decoder_input), dim=1)
                # predict recursively
                else:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        ### adding the other features
                        lumi_feature = target_batch_input[t, :, :]
                        deltat_feature = target_batch_input2[t,:,:]
                        decoder_output = torch.cat((lumi_feature, deltat_feature,decoder_output), dim=1)
                        decoder_input = decoder_output

                # compute the loss
                outputs = outputs.to(self.device)
                data_loss = criterion(outputs, target_batch)
                mu    = torch.tensor(S_cal.mean_[0],  device=self.device)
                sigma = torch.tensor(S_cal.scale_[0], device=self.device)
                target_batch_input2 = self.target_tensor[:, batch: batch + self.batch_size, 5:6]#unscaled
                target_batch_input = self.target_tensor[:, batch: batch + self.batch_size, 3:4]#unscaled
                                # dC, ΔL and time gaps
                C_norm       = outputs.squeeze(-1)                     # [T, B]
                C_phys = C_norm * sigma + mu  
                Ldiff   = target_batch_input.squeeze(-1)          # [T, B]  ← measured ΔL per step
                delta_t = target_batch_input2.squeeze(-1)         # [T, B]  ← Δt per step

                # finite differences
                dC   = C_phys[1:] - C_phys[:-1]                             # [T-1, B]
                Cold = C_phys[:-1]                                     # [T-1, B]
                Δt   = delta_t[:-1]                                # [T-1, B]

                # convert ΔL into rate
                Lrate = Ldiff[:-1] / Δt                            # [T-1, B]
                zero_mask = Ldiff == 0
                # residual of discrete ODE: (ΔC/Δt) + k_dmg*beta*Lrate*C_old = 0
                damage_term = self.k_dmg * Lrate * Cold 
                recovery_term = -self.k_rec * (self.C_inf - Cold)
                # term = torch.where(zero_mask[:-1], recovery_term, damage_term)
                # resid     = dC/Δt + self.k_dmg * Lrate * Cold - self.k_rec*(self.C_inf - Cold)
                resid     = dC/Δt + damage_term + recovery_term
                pinn_loss = criterion(resid, torch.zeros_like(resid))

                # total loss = data loss + λ·physics loss
                loss = data_loss + self.lambda_pinn*pinn_loss
                batch_loss.append(loss.item())
                # backpropagation
                loss.backward()
                optimizer.step()
            epoch_loss = np.mean(batch_loss)
            print('>>>>>> {}/{} Epoch; Loss={}'.format(epoch, self.n_epochs, epoch_loss))
            losses.append(epoch_loss)

            ### we save its loss every print_step
            if epoch % self.print_step == 0:
                 print(f"epoch {epoch}: data={data_loss.item():.4f} phys={pinn_loss.item():.4f}")

        # show_loss(losses)
