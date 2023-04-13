import torch
import torch.nn.functional as F
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.threshold  = args.sparse 
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #self.l0_milestone= [121,128,129,135,200]
        self.milestone=[350,500]
        self.alpha=1
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)
    
    def get_sparse_matrix(self, mask):
        indice = mask.nonzero(as_tuple = True)
        value = mask[indice]
        A = {'indice':indice,'value':value}
        return A
    
    def val_epoch(self, epoch, val_dataloader, fixed_mask = None):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                if fixed_mask is not None:
                    output, mask = self.model.test(data, target, fixed_mask,  teacher_forcing_ratio=0.)
                else:
                    output,mask = self.model(data, target, epoch, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        a = mask
        if mask == None:
            sparse =0
        else:
            sparse = 1-a.nonzero().size(0)/(a.size(0)*a.size(1))
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}, SPARSE:{:.6f}'.format(epoch, val_loss, sparse))
        return val_loss, sparse

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        if epoch in self.milestone:
            self.alpha=self.alpha*1.7
        if epoch>700:
            self.alpha=self.alpha*0.1
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
           
            output, mask = self.model(data, target, epoch, teacher_forcing_ratio=teacher_forcing_ratio)
            
            if mask == None:
                a = self.model.return_supports()
                sparse = 1-a.nonzero().size(0)/(a.size(0)*a.size(1))
                #sparse =0
            else:
                a = mask
                sparse = 1-a.nonzero().size(0)/(a.size(0)*a.size(1))
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            
            #loss = self.loss(output.cuda(), label)+self.model.regularization()
            
            
            if self.args.epochs+self.args.l0epochs>=epoch>self.args.epochs:
                norm_loss = self.model.regularization()
            
                loss = self.loss(output.cuda(), label)+self.alpha*self.model.regularization()
            elif epoch<=self.args.epochs:
                #norm_loss = self.model.l1_norm()
                norm_loss = 0
                loss = self.loss(output.cuda(), label)
            elif epoch>self.args.epochs+self.args.l0epochs:
                norm_loss = self.model.regularization()
                
                loss = self.loss(output.cuda(), label)+self.alpha*self.model.regularization()
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0 and epoch:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} Norm:{:.6f} Sparse:{:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item(), norm_loss, sparse))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
        if 50>epoch>self.args.epochs:
            A = self.get_sparse_matrix(mask) 
            mask_path = 'results/ada_matrix/'+self.args.dataset+'/'+str(epoch)+'.pt'
            torch.save(mask,mask_path)
        else:
            A = None
        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss,A
        
    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        print(self.args.l0epochs)
        for epoch in range(1, self.args.epochs + 1):
            
            #epoch_time = time.time()
            train_epoch_loss,A = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss ,sparse= self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e20:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if self.val_loader == None:
                val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss: 
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                best_optimizer = copy.deepcopy(self.optimizer.state_dict())
        node_embeddings = self.model.node_embeddings.data
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        torch.save(supports,'adj.pt')
        self.model.load_state_dict(best_model)
        self.model.load_state_dict(best_model)
        
        self.optimizer.load_state_dict(best_optimizer)
        self.logger.info('*********************************Loading current best model')
        self.logger.info('*********************************Loading optimizer of the best model')
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        
        for epoch in range(1+self.args.epochs , 1+self.args.epochs +self.args.l0epochs):
            
            train_epoch_loss,A = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss,sparse = self.val_epoch(epoch, val_dataloader)
            
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            #if train_epoch_loss > 1e6:
                #self.logger.warning('Gradient explosion detected. Ending...')
                #break
                
            if val_epoch_loss < best_loss and sparse>0.99: 
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
          
            if best_state == True:
                self.logger.info('*********************************Current l0norm best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            if epoch%20 ==0:
                x_path = 'results/ada_matrix/'+str(epoch)+'.pt'
                torch.save(A,x_path)
       
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
        
        
        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        fixed_mask = self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
        fix_path = 'results/ada_matrix/'+'Sparse_Mat.pt'
        torch.save(A,fix_path)
        return best_model, fixed_mask
    
    
    

        
    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, mask = None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        if mask is None:
            mask = model.sample_weights().detach() 
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output,mask = model.test(data, target, mask,teacher_forcing_ratio=0) 
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.6f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        a = mask
        sparse = 1-a.nonzero().size(0)/(a.size(0)*a.size(1)) 
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, Sparse:{:.4f}".format(
                    mae, rmse, mape*100,sparse))
        #state = model.return_states()
        #state['supports'] = (mask*state['supports']).detach().cpu()
        #a_path = 'results/ada_matrix/'+args.dataset+'/'+'ADJ.pt'
        #state_path = 'results/ada_matrix/'+args.dataset+'/'+'state.pt'
        #model_state_path ='results/'+args.dataset+'.pt'
        #torch.save(a, a_path)
        #torch.save(state, state_path)
        #torch.save(model.state_dict(), model_state_path)
        return mask
        
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
