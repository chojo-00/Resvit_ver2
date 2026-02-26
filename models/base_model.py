import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)



    def save_training_state(self, epoch_label):
        save_filename = '%s_training_state.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        state = {
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sch.state_dict() for sch in self.schedulers],
        }
        torch.save(state, save_path)

    def load_training_state(self, epoch_label):
        save_filename = '%s_training_state.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            state = torch.load(save_path)
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(state['optimizers'][i])
            for i, sch in enumerate(self.schedulers):
                sch.load_state_dict(state['schedulers'][i])
            print('Loaded training state from %s' % save_path)
        else:
            print('No training state found at %s, starting fresh.' % save_path)